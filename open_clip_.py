import torch
from PIL import Image
import open_clip
import matplotlib.pyplot as plt

def compute_similarity(image_path, text_context):
    # Load the model and preprocessing
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
    model.eval()  # Set the model in evaluation mode
    tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')

    # Load and preprocess the image
    image = preprocess(Image.open(image_path)).unsqueeze(0)

    # Text text_context for a single search
    text = tokenizer([text_context])  # Single text_context

    # Feature extraction
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # Normalize features (to unit length) for cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity directly
        similarity = (image_features @ text_features.T).squeeze()

    # Print similarity score
    print(f"Cosine similarity for '{text_context}': {similarity.item():.2f}")

    if __name__ == "__main__":
        # Visualize image with similarity score
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(Image.open(image_path))
        ax.axis('off')  # Hide axis
        ax.set_title(f"Similarity: {similarity.item():.2f} for '{text_context}'")

        # Show the image and similarity
        plt.show()

if __name__ == "__main__":
    image_path = "frame_00013.jpg"
    text_context = "two chairs"
    compute_similarity(image_path, text_context)
