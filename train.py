import numpy as np
import torch
from torch import nn

from dataset.load_dataset import CMPDataset
from evaluate import load
from datasets import load_from_disk
import torchvision.transforms as T
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer
)
from skimage.transform import resize
from transformers import EarlyStoppingCallback


# helper function for ids and labels
id2label = CMPDataset.id2label
label2id = {v: k for k, v in id2label.items()}
num_labels = CMPDataset.num_labels


# dataset loading
dataset_path = "dataset/data/cmp/hf"
ds = load_from_disk(dataset_path)

train_ds, eval_ds, test_ds = ds['train'], ds['eval'], ds['test']


feature_extractor = SegformerImageProcessor()
jitter = T.ColorJitter(brightness=0.25, contrast=0.25,
                       saturation=0.25, hue=0.1)
crop = T.RandomResizedCrop(512)
flip = T.RandomHorizontalFlip(0.5)

def preprocess_images(example_batch, is_train=True):
    """
    Prepares input images and labels for training and evaluation.
    """
    images = [jitter(x) if is_train else x for x in example_batch['pixel_values']]
    labels = [np.array(x, dtype=np.int64) for x in example_batch['label']]

    resized_images = [np.array(image.resize((512, 512))) for image in images]
    resized_labels = [
        resize(label, (512, 512), order=0, preserve_range=True).astype(np.int64) for label in labels
    ]

    inputs = feature_extractor(images=resized_images, return_tensors="pt")
    inputs['labels'] = torch.tensor(resized_labels, dtype=torch.long)

    return inputs

# Apply transformations
train_ds.set_transform(lambda batch: preprocess_images(batch, is_train=True))
eval_ds.set_transform(lambda batch: preprocess_images(batch, is_train=False))
test_ds.set_transform(lambda batch: preprocess_images(batch, is_train=False))

# Load the model pretrained model
pretrained_model_name = "mit-b2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
).to(device)

# Training configuration

epochs = 50
lr = 0.0006
batch_size = 6

training_args = TrainingArguments(
    "your_model_output_dir",
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    eval_strategy="steps",  # Evaluate at regular intervals
    eval_steps=30,          # Adjust frequency of evaluation
    save_strategy="steps",
    save_steps=30,          # Save checkpoints at the same frequency
    logging_steps=10,       # Log loss and metrics more frequently
    eval_accumulation_steps=5,
    lr_scheduler_type="linear",
    load_best_model_at_end=True,
    push_to_hub=False,
    optim='adamw_torch',
    remove_unused_columns=True,
)

metric = load("mean_iou")

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits_tensor, size=labels.shape[-2:], mode="bilinear", align_corners=False
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels, references=labels,
            num_labels=num_labels, ignore_index=0, reduce_labels=False
        )

        return {k: v.tolist()[1:] if isinstance(v, np.ndarray) else v for k, v in metrics.items()}

# Custom Trainer class for loss computation
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").long()

        if len(labels.shape) == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)  # Convert shape if needed

        outputs = model(**inputs)
        logits = nn.functional.interpolate(
            outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )

        loss = nn.CrossEntropyLoss(ignore_index=0)(logits, labels)
        return (loss, outputs) if return_outputs else loss


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)], # Early stopping (uncomment to use)
)

trainer.train()

# Save model and feature extractor
output_dir = "your_model_output_dir"
model.save_pretrained(output_dir)
feature_extractor.save_pretrained(output_dir)