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

class MamiDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, dataset_class: str, annotation_filepaths: dict, img_dir: dict, model_class_or_path: str, batch_size: int, shuffle_train: bool):
        super().__init__()

        # TODO: Separate this into a separate YAML configuration file
        self.dataset_class = globals()[dataset_class]
        self.annotation_filepaths = annotation_filepaths
        self.img_dir = img_dir

        self.batch_size = batch_size
        self.shuffle_train = shuffle_train

        self.batch_size = batch_size
        self.shuffle_train = shuffle_train

        processor = FlavaProcessor.from_pretrained(model_class_or_path)
        self.collate_fn = partial(image_collate_fn_mami, processor=processor)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = self.dataset_class(
                annotation_filepath=self.annotation_filepaths["train"],
                img_dir=self.img_dir["train"]
            )

            self.validate = self.dataset_class(
                annotation_filepath=self.annotation_filepaths["validate"],
                img_dir=self.img_dir["validate"]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.dataset_class(
                annotation_filepath=self.annotation_filepaths["test"],
                img_dir=self.img_dir["test"]
            )

        if stage == "predict" or stage is None:
            self.predict = self.dataset_class(
                annotation_filepath=self.annotation_filepaths["predict"],
                img_dir=self.img_dir["test"]
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
    
    def __init__(self, annotation_filepath, img_dir):
        self.img_annotations = pd.read_csv(annotation_filepath, sep='\t')
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