# AI-Driven Facade Segmentation of Łódź Architecture
## Introduction

### Automated Generation of Facade Segmentation Maps of Łódź Architecture Using Artificial Intelligence

## Installation

First, clone the repository by running the following command:
```bash
git clone https://github.com/Koorbik/ai-facade-segmentation-model.git
```

Then install the specific version of PyTorch:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install the rest of the requirements:
```bash
pip3 install -r requirements.txt
```

## Best Model Directory (`best_model_dir`)
This directory contains the best model for the facade segmentation task. The model is trained on the CMP dataset from the Lodz University of Technology using the following hyperparameters:
- **Batch size:** 6
- **Learning rate:** 0.0006
- **Epochs:** 50
- **Optimizer:** `adamw_torch`
- **Learning rate scheduler:** `linear`

The model is a fine-tuned version of NVIDIA’s **SegFormer mit-b2** and was trained on a dataset combining the **CMP-Facades dataset** with approximately **60 images of Łódź facades**. The training was conducted using a **single RTX 4070Ti GPU**.

### Model Performance on Validation Set:
- **mIoU:** 0.58 (due to a small dataset; performance is expected to improve with more data)
- **Overall Accuracy:** 0.80

While the model is relatively effective, it has limitations in segmenting some facades and is somewhat prone to overfitting due to the limited dataset.

## Data Preparation

The data is prepared as follows:
- The dataset is divided into **training (80%)**, **validation (10%)**, and **test (10%)** sets (see `datasetPreparer.py`).
- Each set contains both images (`.jpg` files) and corresponding segmentation masks (`.png` files).
- The masks for the Łódź facades were manually annotated using the **CVAT tool** to closely resemble the CMP-Facades dataset, though they are not perfect.

### Mask Labels:
```
0: "unknown",
1: "background",
2: "facade",
3: "window",
4: "door",
5: "cornice",
6: "sill",
7: "balcony",
8: "blind",
9: "pillar",
10: "deco",
11: "molding",
12: "shop"
```

### CVAT Annotation Colors:
Each label has a unique color used for annotation:
```json
[
  {"name": "window", "color": "#0055ff"},
  {"name": "door", "color": "#00aaff"},
  {"name": "facade", "color": "#0000ff"},
  {"name": "cornice", "color": "#00ffff"},
  {"name": "sill", "color": "#55ffaa"},
  {"name": "balcony", "color": "#aa0000"},
  {"name": "blind", "color": "#aaff55"},
  {"name": "background", "color": "#0000aa"},
  {"name": "deco", "color": "#ff5500"},
  {"name": "shop", "color": "#ffff00"},
  {"name": "molding", "color": "#ffaa00"},
  {"name": "pillar", "color": "#ff0000"},
  {"name": "unknown", "color": "#000000"}
]
```

### Preprocessing Steps:
The CMP-Facades dataset used a custom MATLAB script (not available) for annotation, meaning reprocessing is required:
1. **Convert Annotated Masks to Grayscale**
   - Add annotated masks (with unique colors) to the `masks_to_reprocess` directory.
   - Run `reprocessMasks.py` to process and save masks in the `reprocessed_masks` directory.
2. **Prepare the Dataset for Training**
   - Run `datasetPreparer.py` (ensure the correct input directory contains both images and masks).
   - The script splits data into training, validation, and test sets in the `data` directory.
3. **Load the Dataset**
   - Use `load_dataset.py` from the `dataset` directory.

## Training

To train the model, simply run:
```bash
python train.py
```
You can adjust hyperparameters in `train.py`. By default, it fine-tunes the **SegFormer mit-b2** model from NVIDIA and saves the best model in the specified directory.

## Inference

To perform inference on images from the train, validation, or test sets:
1. Edit and run `inference.py`.
2. The script will display the images alongside their ground truth and predicted segmentation masks.

## Testing

To test the model on the test set and generate segmentation masks, run:
```bash
python test.py
```
The generated segmentation masks will be saved in the `AI-generated-segmentation-masks` directory.

Some example masks have already been provided.

## Author Information
Hubert Szadkowski ([Koorbik](https://github.com/Koorbik)) - Computer Science Student at the Łódz University of Technology