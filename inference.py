import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from dataset.load_dataset import CMPDataset

# Helper variables
unique_colors = CMPDataset.unique_colors
id2label = CMPDataset.id2label
label2id = {v: k for k, v in id2label.items()}

# Paths
model_path = "best_model_dir"
dataset_path = "dataset/data/cmp/hf"

# Load model and feature extractor
model = SegformerForSemanticSegmentation.from_pretrained(model_path).eval()
feature_extractor = SegformerImageProcessor.from_pretrained(model_path)

# Load dataset for visualization
from datasets import load_from_disk

ds = load_from_disk(dataset_path)
train_ds = ds['train']
test_ds = ds['test']
eval_ds = ds['eval']


# Inference function
def predict_segmentation(image, model, feature_extractor):
    """
    Perform inference on a single image using the trained model.
    """
    # Ensure the image is in the correct format
    if len(image.shape) == 3 and image.shape[0] == 3:
        # Convert from (channels, height, width) to (height, width, channels)
        image = image.transpose((1, 2, 0))
    elif len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must have shape (height, width, 3) or (3, height, width).")

    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits and upsample to original size
    logits = outputs.logits  # Shape: (batch_size, num_labels, h/4, w/4)
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=image.shape[:2], mode='bilinear', align_corners=False
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()  # Shape: (height, width)
    return pred_seg


# Visualization function
def get_seg_overlay(image, seg):
    """
    Create an overlay of the image with the segmentation mask.
    """
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    palette = np.array(unique_colors)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    # Blend the image and the segmentation mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    return img.astype(np.uint8)


def visualize_prediction(image, pred_seg, gt_seg=None):
    """
    Visualize the prediction and optionally the ground truth.
    """
    pred_img = get_seg_overlay(image, pred_seg)
    fig, axs = plt.subplots(1, 2 if gt_seg is not None else 1, figsize=(15, 15))

    axs[0].imshow(pred_img)
    axs[0].set_title("Prediction", fontsize=20)
    axs[0].axis("off")

    if gt_seg is not None:
        gt_img = get_seg_overlay(image, gt_seg)
        axs[1].imshow(gt_img)
        axs[1].set_title("Ground Truth", fontsize=20)
        axs[1].axis("off")

    plt.show()


# Inference on test sample
img_idx = 1
sample = test_ds[img_idx]
image = np.array(sample['pixel_values'])  # Input image
gt_seg = np.array(sample['label'])  # Ground truth mask

# Predict
pred_seg = predict_segmentation(image, model, feature_extractor)

# Visualize
visualize_prediction(image, pred_seg, gt_seg)
