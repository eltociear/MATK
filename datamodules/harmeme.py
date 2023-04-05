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

   
class HarMemeDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, dataset_class: str, annotation_filepaths: dict, img_dir: str, model_class_or_path: str, batch_size: int, shuffle_train: bool):
        super().__init__()

        # TODO: Separate this into a separate YAML configuration file
        self.dataset_class = globals()[dataset_class]
        self.annotation_filepaths = annotation_filepaths
        self.img_dir = img_dir

        self.batch_size = batch_size
        self.shuffle_train = shuffle_train

        processor = FlavaProcessor.from_pretrained(model_class_or_path)
        self.collate_fn = partial(image_collate_fn_harmeme, processor=processor)


    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = self.dataset_class(
                annotation_filepath=self.annotation_filepaths["train"],
                img_dir=self.img_dir
            )

            self.validate = self.dataset_class(
                annotation_filepath=self.annotation_filepaths["validate"],
                img_dir=self.img_dir
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.dataset_class(
                annotation_filepath=self.annotation_filepaths["test"],
                img_dir=self.img_dir
            )

        if stage == "predict" or stage is None:
            self.predict = self.dataset_class(
                annotation_filepath=self.annotation_filepaths["predict"],
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

    def __init__(self, annotation_filepath, img_dir):
        self.img_annotations = pd.read_json(annotation_filepath, lines=True)
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