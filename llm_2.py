from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset
import torch

# Load dataset
data = [
    {"input": '{"objects": [{"id": 1, "name": "table", "position": [0, 0, 0]}, {"id": 2, "name": "chair", "position": [1, 0, 0]}]}', 
     "output": '"table" → "next to" → "chair"'}
]
dataset = Dataset.from_dict(data)

# Load T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Tokenize the inputs and outputs
def tokenize_function(examples):
    return tokenizer(examples['input'], truncation=True, padding="max_length", max_length=512)

def tokenize_output(examples):
    return tokenizer(examples['output'], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.map(tokenize_output, batched=True)

# Set up training arguments
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',         
    evaluation_strategy="epoch",     
    learning_rate=2e-5,
    per_device_train_batch_size=16, 
    num_train_epochs=3,
    weight_decay=0.01,
)

# Train the model
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=dataset,         
)

trainer.train()