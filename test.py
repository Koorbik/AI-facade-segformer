import os
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from dataset.load_dataset import CMPDataset

# Load the model and feature extractor
output_dir = "best_model_dir"
model = SegformerForSemanticSegmentation.from_pretrained(output_dir)
feature_extractor = SegformerImageProcessor.from_pretrained(output_dir)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Helper function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    resized_image = image.resize((512, 512))  # Resize to match training dimensions
    inputs = feature_extractor(images=resized_image, return_tensors="pt")
    return inputs, np.array(image)

# Function to save the predicted mask
def save_segmentation_mask(original_image, predicted_mask, unique_colors, output_path):
    h, w, _ = original_image.shape
    color_seg = np.zeros((h, w, 3), dtype=np.uint8)
    palette = np.array(unique_colors)

    for label, color in enumerate(palette):
        color_seg[predicted_mask == label, :] = color

    # Save the segmentation mask as an image
    segmentation_image = Image.fromarray(color_seg)
    segmentation_image.save(output_path)

# Inference function for batch processing
def batch_predict_and_save(test_folder, output_folder, unique_colors):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Process all files in the test folder
    for filename in os.listdir(test_folder):
        file_path = os.path.join(test_folder, filename)
        if os.path.isfile(file_path) and file_path.endswith((".jpg")):
            # Preprocess the image
            inputs, original_image = preprocess_image(file_path)

            # Move inputs to the same device as the model
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Perform inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                logits = torch.nn.functional.interpolate(
                    logits,
                    size=original_image.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                ).argmax(dim=1)
                predicted_mask = logits[0].cpu().numpy()

            # Save the predicted mask
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_segmentation.png")
            save_segmentation_mask(original_image, predicted_mask, unique_colors, output_path)
            print(f"Saved segmentation mask for {filename} to {output_path}")

# Define unique colors for visualization (from CMPDataset)
unique_colors = CMPDataset.unique_colors

# Paths
test_folder = "dataWithCMP/test"
output_folder = "AI-generated-segmentation-masks"


batch_predict_and_save(test_folder, output_folder, unique_colors)
