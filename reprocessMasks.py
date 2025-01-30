import os
import numpy as np
from PIL import Image


def reprocess_mask(input_path, output_path, unique_colors):
    """
    Reprocess a mask by mapping RGB pixel values to class ids and save the result.

    Args:
        input_path (str): Path to the input mask image.
        output_path (str): Path to save the reprocessed mask.
        unique_colors (list): List of RGB colors representing classes.
    """
    # Load the segmentation image as an RGB NumPy array
    segmentation_image = np.array(Image.open(input_path).convert('RGB'))

    # Map RGB values to class ids
    color_to_class = {tuple(color): i for i, color in enumerate(unique_colors)}
    h, w, _ = segmentation_image.shape
    reshaped_image = segmentation_image.reshape(-1, 3)
    output_flat = np.array([color_to_class[tuple(pixel)] for pixel in reshaped_image], dtype=np.uint8)
    remapped_mask = output_flat.reshape(h, w)

    # Save the remapped mask
    Image.fromarray(remapped_mask).save(output_path)


def reprocess_all_masks(input_dir, output_dir, unique_colors):
    """
    Reprocess all masks in a directory by mapping RGB pixel values to class ids.

    Args:
        input_dir (str): Path to the directory containing the original masks.
        output_dir (str): Path to save the reprocessed masks.
        unique_colors (list): List of RGB colors representing classes.
    """
    os.makedirs(output_dir, exist_ok=True)
    mask_files = [f for f in os.listdir(input_dir) if f.endswith(".png")]

    for file_name in mask_files:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        try:
            reprocess_mask(input_path, output_path, unique_colors)
            print(f"Reprocessed and saved: {output_path}")
        except KeyError as e:
            print(f"Error processing {file_name}: Unrecognized color {e}")


if __name__ == "__main__":
    # Define the path to the input directory containing masks and the output directory
    input_dir = "masks_to_reprocess"  # Replace with your input directory path
    output_dir = "reprocessed_masks"  # Replace with your output directory path

    # Define the unique colors for the classes
    unique_colors = [
        [0, 0, 0],
        [0, 0, 170],
        [0, 0, 255],
        [0, 85, 255],
        [0, 170, 255],
        [0, 255, 255],
        [85, 255, 170],
        [170, 0, 0],
        [170, 255, 85],
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0]
    ]

    # Call the function to reprocess all masks
    reprocess_all_masks(input_dir, output_dir, unique_colors)
