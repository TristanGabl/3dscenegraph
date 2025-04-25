
from relation_net import load_model, generate_edge_relationship
import torch

import numpy as np


# Initialize the model
model, name2idx, label2idx, idx2label = load_model(
    model_path='dataset/relation_model.pth',
    
    device='cpu'  # Change to 'cuda' if you have a GPU
)

# Encode categorical features as numerical values

src = {'name': 'chair', 'x': 0.0, 'y': 0.0, 'z': 0.5, 'size_x': 0.5, 'size_y': 0.5, 'size_z': 1}
tgt = {'name': 'wall', 'x': 3.0, 'y': 1.0, 'z': 0.5, 'size_x': 0.5, 'size_y': 5, 'size_z': 5}


# Convert prediction index to label
pred_label = generate_edge_relationship(src, tgt, model, name2idx, label2idx, idx2label)
print(f"Predicted label: {pred_label}")