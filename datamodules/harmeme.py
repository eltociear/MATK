import torch
import numpy as np
import pandas as pd
import lightning.pytorch as pl

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

from functools import partial
from torchvision.transforms import ToTensor

from typing import Optional
from transformers import FlavaProcessor

from .utils import image_collate_fn_harmeme

def get_dataset_attributes(dataset_name: str):
    img_dir = "/mnt/sdb/aditi/mmf/data/datasets/memes/defaults/images/"

    if dataset_name == "harmeme":
        return HarMemeDataset, {
            "train": "/mnt/sdb/aditi/mmf/data/datasets/memes/defaults/annotations/train.jsonl",
            "validate": "/mnt/sdb/aditi/mmf/data/datasets/memes/defaults/annotations/val.jsonl",
            "test": "/mnt/sdb/aditi/mmf/data/datasets/memes/defaults/annotations/test.jsonl"
        }, img_dir
   
class HarMemeDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, dataset_name, model_class_or_path, batch_size, shuffle_train, **kwargs):
        super().__init__()

        # TODO: Separate this into a separate YAML configuration file
        self.dataset_class, self.annotations_fp, self.img_dir = get_dataset_attributes(dataset_name)

        self.batch_size = batch_size
        self.shuffle_train = shuffle_train

        processor = FlavaProcessor.from_pretrained(model_class_or_path)
        self.collate_fn = partial(image_collate_fn_harmeme, processor=processor)


    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = self.dataset_class(
                annotations_file=self.annotations_fp["train"],
                img_dir=self.img_dir
            )

            self.validate = self.dataset_class(
                annotations_file=self.annotations_fp["validate"],
                img_dir=self.img_dir
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.dataset_class(
                annotations_file=self.annotations_fp["test"],
                img_dir=self.img_dir
            )

        if stage == "predict" or stage is None:
            self.predict = self.dataset_class(
                annotations_file=self.annotations_fp["predict"],
                img_dir=self.img_dir
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)


class HarMemeDataset(Dataset):

    def __init__(self, annotations_file, img_dir):
        self.img_annotations = pd.read_json(annotations_file, lines=True)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_annotations)

    def __getitem__(self, idx):
        img_id = self.img_annotations.loc[idx, 'id']
        image = self.img_annotations.loc[idx, 'image']
        text = self.img_annotations.loc[idx, 'text']
        labels = self.img_annotations.loc[idx, 'labels']
        img_path = self.img_dir + image
        
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img = img.convert("RGB") if img.mode != "RGB" else img
        return {
            'id': img_id,
            'text': text, 
            'image': np.array(img),
            'labels': labels
        }