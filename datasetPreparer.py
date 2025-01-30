import os
import shutil
import random

# Directories
source_dir = "reprocessed_masks"
output_dir = "data"

train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "eval")
test_dir = os.path.join(output_dir, "test")

# Create output directories
for directory in [train_dir, val_dir, test_dir]:
    os.makedirs(directory, exist_ok=True)

# Collect file names
all_files = os.listdir(source_dir)
image_files = {f for f in all_files if f.endswith(".jpg")}
mask_files = {f for f in all_files if f.endswith(".png")}

# Match images and masks based on filenames
matched_files = sorted(
    {os.path.splitext(f)[0] for f in image_files} & {os.path.splitext(f)[0] for f in mask_files}
)

# Shuffle the data for randomness
random.seed(42)  # For reproducibility
random.shuffle(matched_files)

# Split into train (80%), val (10%), and test (10%)
num_total = len(matched_files)
num_train = int(0.8 * num_total)
num_val = int(0.1 * num_total)

train_files = matched_files[:num_train]
val_files = matched_files[num_train:num_train + num_val]
test_files = matched_files[num_train + num_val:]

# Function to copy files
def copy_files(file_list, source_dir, target_dir):
    for base_name in file_list:
        image_file = base_name + ".jpg"
        mask_file = base_name + ".png"

        image_src = os.path.join(source_dir, image_file)
        mask_src = os.path.join(source_dir, mask_file)

        image_dst = os.path.join(target_dir, image_file)
        mask_dst = os.path.join(target_dir, mask_file)

        if os.path.exists(image_src) and os.path.exists(mask_src):
            shutil.copy2(image_src, image_dst)
            shutil.copy2(mask_src, mask_dst)
            print(f"Copied {image_file} and {mask_file} to {target_dir}")
        else:
            print(f"Missing file(s) for base {base_name}: {image_src}, {mask_src}")

# Copy the files
copy_files(train_files, source_dir, train_dir)
copy_files(val_files, source_dir, val_dir)
copy_files(test_files, source_dir, test_dir)

print("Dataset split complete.")


source_dir = "reprocessed_masks"
print(f"Source directory: {source_dir}")
print("Files in source directory:", os.listdir(source_dir))