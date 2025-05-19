import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup
)

class EdgeDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 tokenizer: GPT2Tokenizer,
                 input_feats_numeric: list[str],
                 input_feats_text: list[str],
                 gt_col: str,
                 max_length: int = 64):

        # numeric features → float tensor
        self.feats = torch.tensor(df[input_feats_numeric].values,
                                  dtype=torch.float32)

        # prompt text: join your text‐fields (e.g. "objA, objB")
        self.texts = (df[input_feats_text]
                 .astype(str)
                 .agg(lambda x: f"Relation between {x[0]} and {x[1]} is:", axis=1)
                 .tolist())

        # ground‐truth relation (sentence) as strings
        self.gt_texts = df[gt_col].astype(str).tolist()

        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        feat       = self.feats[idx]      # (9,)
        prompt     = self.texts[idx]      # e.g. "cup, table"
        relation   = self.gt_texts[idx]   # e.g. "is on top of"

        # tokenize the prompt
        enc_prompt = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        # tokenize the target relation
        enc_gt = self.tokenizer(
            relation,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids":      enc_prompt["input_ids"].squeeze(0),       # (L,)
            "attention_mask": enc_prompt["attention_mask"].squeeze(0),  # (L,)
            "features":       feat,                                     # (9,)
            "labels":         enc_gt["input_ids"].squeeze(0),           # (L,)
        }


def collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "features":       torch.stack([b["features"]       for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
    }

# 2) Extend GPT-2 to ingest numeric features -------------------------------
class GPT2WithFeats(GPT2LMHeadModel):
    def __init__(self, config, n_feats):
        super().__init__(config)
        self.feat_proj = nn.Linear(n_feats, config.hidden_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        features=None,
        labels=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,                  # <— catch everything else
    ):
        # 1) get embeddings (either via ids or passed in)
        if inputs_embeds is None:
            inputs_embeds = self.transformer.wte(input_ids)

        # 2) project and add your numeric features into the first token
        feat_emb = self.feat_proj(features)
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[:, 0, :] = inputs_embeds[:, 0, :] + feat_emb

        # 3) forward through GPT-2, handing off any extra flags
        return super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
            **kwargs,               # <— let return_dict, etc. through
        )

# 3) Setup everything -------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # so padding is valid
tokenizer.padding_side = "right"  # so that the features are aligned with the text

config = GPT2Config.from_pretrained('gpt2')
model  = GPT2WithFeats(config, n_feats=10) # 10 numeric features
model.load_state_dict(GPT2LMHeadModel.from_pretrained('gpt2').state_dict(), strict=False)
model.to(device)

# assume you have a pandas DataFrame `df` with columns num_feats + 'sentence'
input_feats_numeric = ['dx','dy','dz','dist',
             'size_x1','size_y1','size_z1',
             'size_x2','size_y2','size_z2']
input_feats_text = ['src_name','tgt_name']
df = pd.read_csv('dataset/processed_relationship_data.csv')
gt = 'relation'
dataset = EdgeDataset(df[:100], tokenizer, input_feats_numeric, input_feats_text, gt, max_length=64)
loader  = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# optimizer & scheduler
epochs = 1
total_steps = len(loader) * epochs
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)


# 4) Training loop ----------------------------------------------------------
if (False):  # set to False to skip training
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, batch in enumerate(loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            feats          = batch["features"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                features=feats,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            print(f"[Epoch {epoch+1:02d}]  batch {i+1}/{len(loader)}  loss: {loss.item():.4f}")

        avg = epoch_loss / len(loader)
        print(f"[Epoch {epoch+1:02d}]  avg loss: {avg:.4f}")

    # 5) Save your fine-tuned model ---------------------------------------------
    model.save_pretrained('./gpt2-scenegraph-finetuned')
    tokenizer.save_pretrained('./gpt2-scenegraph-finetuned')

else:
    # 5) Load your fine-tuned model ---------------------------------------------
    model = GPT2WithFeats.from_pretrained('./gpt2-scenegraph-finetuned', n_feats=10)
    tokenizer = GPT2Tokenizer.from_pretrained('./gpt2-scenegraph-finetuned')
    model.to(device)
    # 6) Inference --------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            feats          = batch["features"].to(device)

            # Decode input prompts to show what objects are being fed
            input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            # Instead of feeding "objA, objB", create a more explicit prompt:
            # Example: "Relation between cup and table is"
            explicit_prompts = [
                f"Relation between {text} is"
                for text in input_texts
            ]

            # Tokenize these explicit prompts (left padded)
            enc_prompts = tokenizer(
                explicit_prompts,
                truncation=True,
                padding="max_length",
                max_length=64,
                return_tensors="pt"
            )
            input_ids_explicit = enc_prompts["input_ids"].to(device)
            attention_mask_explicit = enc_prompts["attention_mask"].to(device)

            # Generate relation continuations after the prompt
            outputs = model.generate(
                input_ids=input_ids_explicit,
                attention_mask=attention_mask_explicit,
                features=feats,
                max_new_tokens=32,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id  # avoid warning about no pad token
            )

            # Decode generated sequences
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Print prompt and generated relation (remove the prompt part for clarity)
            for prompt, gen_text in zip(explicit_prompts, generated_texts):
                relation = gen_text[len(prompt):].strip()  # get generated part only
                print(f"Input objects: {prompt[len('Relation between '):-len(' is')]}")
                print(f"Predicted relation: {relation}")
                print("-----")

            break  # just do first batch for demo
