import torch
import numpy as np
import pandas as pd
import lightning.pytorch as pl

from torch.utils.data import DataLoader, Dataset
from PIL import Image

from functools import partial
from torchvision.transforms import ToTensor

from typing import Optional
from transformers import FlavaProcessor
from .utils import image_collate_fn_mami

def get_dataset_attributes(dataset_name: str):
    train_img_dir = "/mnt/sdb/aditi/MAMI/training/TRAINING/"
    val_img_dir = "/mnt/sdb/aditi/MAMI/trial/Users/fersiniel/Desktop/MAMI - TO LABEL/TRIAL DATASET/"
    test_img_dir = "/mnt/sdb/aditi/MAMI/test/test/"


    if dataset_name == "mami":
        return MamiDataset, {
            "train": "/mnt/sdb/aditi/MAMI/training/TRAINING/training.csv",
            "validate": "/mnt/sdb/aditi/MAMI/trial/Users/fersiniel/Desktop/MAMI - TO LABEL/TRIAL DATASET/trial.csv",
            "test": "/mnt/sdb/aditi/MAMI/test/test/Test.csv",
        }, train_img_dir, val_img_dir, test_img_dir

class MamiDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, dataset_name, model_class_or_path, batch_size, shuffle_train, **kwargs):
        super().__init__()

        # TODO: Separate this into a separate YAML configuration file
        self.dataset_class, self.annotations_fp, self.img_dir, self.val_img_dir, self. test_img_dir = get_dataset_attributes(dataset_name)

        self.batch_size = batch_size
        self.shuffle_train = shuffle_train

        processor = FlavaProcessor.from_pretrained(model_class_or_path)
        self.collate_fn = partial(image_collate_fn_mami, processor=processor)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = self.dataset_class(
                annotations_file=self.annotations_fp["train"],
                img_dir=self.img_dir
            )

            self.validate = self.dataset_class(
                annotations_file=self.annotations_fp["validate"],
                img_dir=self.val_img_dir
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.dataset_class(
                annotations_file=self.annotations_fp["test"],
                img_dir=self.test_img_dir
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

class MamiDataset(Dataset):
    
    def __init__(self, annotations_file, img_dir):
        self.img_annotations = pd.read_csv(annotations_file, sep='\t')
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_annotations)

    def __getitem__(self, idx):

        file_name = self.img_annotations.loc[idx, 'file_name']
        misogynous = self.img_annotations.loc[idx, "misogynous"]
        shaming = self.img_annotations.loc[idx, "shaming"]
        stereotype = self.img_annotations.loc[idx, "stereotype"]
        objectification = self.img_annotations.loc[idx, "objectification"]
        violence = self.img_annotations.loc[idx, "violence"]

        text = self.img_annotations.loc[idx, "Text Transcription"]

        img_path = self.img_dir + file_name
        
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img = img.convert("RGB") if img.mode != "RGB" else img

        return {
            'file_name': file_name,
            'misogynous': misogynous, 
            'shaming': shaming,
            'stereotype': stereotype,
            'objectification': objectification,
            'violence': violence,
            'image': np.array(img),
            'text': text
        }