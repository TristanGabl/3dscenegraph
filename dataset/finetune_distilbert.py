#!/usr/bin/env python3
import json
import torch
from torch.quantization import quantize_dynamic
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

def main_training():
    # 1. Load your JSONL dataset
    #    Expects each line: {"prompt": "...", "completion": " label"}
    ds = load_dataset("json", data_files={"train": "3RScan/relationship_training_data.jsonl"}, split="train")

    # 2. Build label mappings
    labels = sorted(set(ds["completion"]))
    label2id = {lbl: idx for idx, lbl in enumerate(labels)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}

    print(f"Found {len(labels)} distinct labels.")

    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # 4. Preprocessing function: tokenize and map labels to IDs
    def preprocess_function(examples):
        tokens = tokenizer(
            examples["prompt"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
        tokens["labels"] = [label2id[c] for c in examples["completion"]]
        return tokens

    tokenized_ds = ds.map(
        preprocess_function,
        batched=True,
        remove_columns=ds.column_names,
        desc="Tokenizing dataset"
    )

    # 5. Shuffle and split into train/eval (90/10)
    tokenized_ds = tokenized_ds.shuffle(seed=42)
    # train_size = int(0.9 * len(tokenized_ds))
    # train_ds = tokenized_ds.select(range(train_size))
    # eval_ds  = tokenized_ds.select(range(train_size, len(tokenized_ds)))

    # Use a tiny amount of data for testing
    train_size = int(0.001 * len(tokenized_ds))  # 1% of the dataset
    train_ds = tokenized_ds.select(range(train_size))
    eval_ds = tokenized_ds.select(range(train_size, train_size + int(0.001 * len(tokenized_ds))))

    # 6. Load DistilBERT for sequence classification
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    # 7. Training arguments
    training_args = TrainingArguments(
        output_dir="./distilbert-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=5000,
        save_steps=5000,
        save_total_limit=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="./logs",
        logging_steps=1000,
    )

    # 8. Compute accuracy metric
    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        acc = (preds == labels).astype(float).mean().item()
        return {"accuracy": acc}

    # 9. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 10. Train
    trainer.train()

    # 11. Save the best model
    trainer.save_model("./distilbert-finetuned-final")

    # 12. Dynamic quantization for faster CPU inference
    quantized_model = quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), "distilbert-quantized.pt")

    # 13. Save label mappings for inference
    with open("label2id.json", "w") as f:
        json.dump(label2id, f, indent=2)
    with open("id2label.json", "w") as f:
        json.dump(id2label, f, indent=2)

    print("✅ Training complete.")
    print("Quantized weights saved to distilbert-quantized.pt")
    print("Label mappings saved to label2id.json / id2label.json")

def main_inference():

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Load the base model and apply dynamic quantization
    model = DistilBertForSequenceClassification.from_pretrained(
        "./distilbert-finetuned-final",
        id2label=json.load(open("id2label.json")),
        label2id=json.load(open("label2id.json"))
    )
    model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    # Load quantized weights
    state_dict = torch.load("distilbert-quantized.pt")
    model.load_state_dict(state_dict)

    # Set model to evaluation mode
    model.to("cpu")
    model.eval()

    # Prepare input
    text = "towel has position offset dx = -0.1m, dy = -0.1m, dz = -0.5m, to bath cabinet."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Predict
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = logits.argmax(-1).item()

    print("Predicted label:", model.config.id2label[f'{pred}'])

if __name__ == "__main__":
    main_training()
    main_inference()




