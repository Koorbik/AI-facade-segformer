import os
from PIL import Image
import numpy as np
import torch
from datasets import Dataset, DatasetDict


class CMPDataset(torch.utils.data.Dataset):
    num_labels = 13

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

    id2label = {
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
    }

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        self.lbl_paths = []
        self.label2id = {lbl: id for id, lbl in self.id2label.items()}

        # Collect file paths
        for file in sorted(os.listdir(root_dir)):
            if file.endswith(".jpg"):
                self.img_paths.append(os.path.join(root_dir, file))
            elif file.endswith(".png"):
                self.lbl_paths.append(os.path.join(root_dir, file))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        lbl_path = self.lbl_paths[idx]

        img = Image.open(img_path).convert("RGB").copy()
        mask = Image.open(lbl_path).convert('P').copy()

        if self.transform:
            img, mask = self.transform(img), self.transform(mask)

        return img, mask

    def parse_segmentation_image(self, segmentation_image):
        """
        Map RGB pixel values in the segmentation image to class ids.
        """
        color_to_class = {tuple(color): i for i, color in enumerate(self.unique_colors)}
        output = np.zeros((segmentation_image.shape[0], segmentation_image.shape[1]), dtype=np.uint8)

        for i in range(segmentation_image.shape[0]):
            for j in range(segmentation_image.shape[1]):
                output[i, j] = color_to_class[tuple(segmentation_image[i, j])]

        return output


if __name__ == '__main__':
    cmp_ds_train = CMPDataset(root_dir='../dataWithCMP/train')
    cmp_ds_eval = CMPDataset(root_dir='../dataWithCMP/eval')
    cmp_ds_test = CMPDataset(root_dir='../dataWithCMP/test')
    id2label = CMPDataset.id2label
    label2id = {v: k for k, v in id2label.items()}
    num_labels = CMPDataset.num_labels

    imgs = []
    lbls = []
    for img, lbl in cmp_ds_train:
        imgs.append(img)
        lbls.append(lbl)
    train_ds = Dataset.from_dict({"pixel_values": imgs, "label": lbls})

    imgs = []
    lbls = []
    for img, lbl in cmp_ds_eval:
        imgs.append(img)
        lbls.append(lbl)
    eval_ds = Dataset.from_dict({"pixel_values": imgs, "label": lbls})

    imgs = []
    lbls = []
    for img, lbl in cmp_ds_test:
        imgs.append(img)
        lbls.append(lbl)
    test_ds = Dataset.from_dict({"pixel_values": imgs, "label": lbls})

    ds = DatasetDict({
        'train': train_ds,
        'eval': eval_ds,
        'test': test_ds
    })

    # Save the dataset to disk
    ds.save_to_disk('data/cmp/hf')
