import os
import tqdm
import pickle as pkl
import numpy as np

import lightning.pytorch as pl


from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# from .fhm import VisionLanguageDataset as FHMChallengeDataset
from .datasets.fhm_finegrained import LanguageDataset as FHMFGTextDataset
from .datasets.fhm_finegrained import VLImagesDataset as FHMFGImagesDataset
from .datasets.fhm_finegrained import VLFeaturesDataset as FHMFGFeaturesDataset

from datamodules.collators import get_collator

from typing import List, Optional

class VLFeaturesDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(
        self,
        dataset_cls: str,
        annotation_filepaths: dict,
        tokenizer_class_or_path: str,
        feats_dir: str,
        batch_size: int,
        shuffle_train: bool,
        task: str,
        labels: List[str]
    ):
        super().__init__()

        self.annotation_filepaths = annotation_filepaths
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.task = task
        self.labels = labels
        
        self.feats_dict = self._load_feats_frcnn(feats_dir)
        self.collate_fn = get_collator(
            tokenizer_class_or_path, 
            labels=labels,
            feats_dict=self.feats_dict
        )
        self.dataset_cls = globals()[dataset_cls]
    
    def _load_feats_frcnn(self, feats_dir: str):
        feats_dict = {}

        files = [
            x for x in os.listdir(feats_dir)
            if ".pkl" in x
        ]
        for filename in tqdm.tqdm(files, desc='Loading features'):
            filepath = os.path.join(feats_dir, filename)
            
            filename, _ = os.path.splitext(filename)
            # FHM workaround
            if filename[0] == "0":
                filename = filename[1:]

            with open(filepath, "rb") as f:
                feats_dict[filename] = pkl.load(f)

        return feats_dict

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["train"],
                task=self.task,
                labels=self.labels
            )

            self.validate = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["validate"],
                task=self.task,
                labels=self.labels
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["test"],
                task=self.task,
                labels=self.labels
            )

        if stage == "predict" or stage is None:
            self.predict = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["predict"],
                task=self.task,
                labels=self.labels
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)

class VLImagesDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(
        self,
        dataset_cls: str,
        annotation_filepaths: dict,
        image_dir: str,
        tokenizer_class_or_path: str,
        frcnn_class_or_path: str,
        batch_size: int,
        shuffle_train: bool,
        task: str,
        labels: List[str]
    ):
        super().__init__()

        self.annotation_filepaths = annotation_filepaths
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.task = task
        self.labels = labels

        self.collate_fn = get_collator(
            tokenizer_class_or_path, 
            labels=labels, 
            frcnn_class_or_path=frcnn_class_or_path
        )
        self.dataset_cls = globals()[dataset_cls]

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["train"],
                image_dir=self.image_dir,
                task=self.task,
                labels=self.labels
            )

            self.validate = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["validate"],
                image_dir=self.image_dir,
                task=self.task,
                labels=self.labels
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["test"],
                image_dir=self.image_dir,
                task=self.task,
                labels=self.labels
            )

        if stage == "predict" or stage is None:
            self.predict = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["predict"],
                image_dir=self.image_dir,
                task=self.task,
                labels=self.labels
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)


class LanguageDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(
        self,
        dataset_cls: str,
        annotation_filepaths: dict,
        tokenizer_class_or_path: str,
        auxiliary_dicts: dict,
        input_template: str,
        output_template: str,
        label2word: dict,
        batch_size: int,
        shuffle_train: bool,
        task: str,
        labels: List[str]
    ):
        super().__init__()

        # ensure that word for each label is a single token.
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_class_or_path, use_fast=False)
        for word in label2word.values():
            encoded = tokenizer.encode(word, add_special_tokens=False)
            assert len(encoded) == 1

        self.annotation_filepaths = annotation_filepaths
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.auxiliary_dicts = auxiliary_dicts
        self.input_template = input_template
        self.output_template = output_template
        self.label2word = label2word
        self.task = task
        self.labels = labels
        self.collate_fn = get_collator(tokenizer_class_or_path, labels=labels)

        self.dataset_cls = globals()[dataset_cls]

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["train"],
                auxiliary_dicts=self.auxiliary_dicts,
                input_template=self.input_template,
                output_template=self.output_template,
                label2word=self.label2word,
                task=self.task,
                labels=self.labels
            )

            self.validate = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["validate"],
                auxiliary_dicts=self.auxiliary_dicts,
                input_template=self.input_template,
                output_template=self.output_template,
                label2word=self.label2word,
                task=self.task,
                labels=self.labels
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["test"],
                auxiliary_dicts=self.auxiliary_dicts,
                input_template=self.input_template,
                output_template=self.output_template,
                label2word=self.label2word,
                task=self.task,
                labels=self.labels
            )

        if stage == "predict" or stage is None:
            self.predict = self.dataset_cls(
                annotation_filepath=self.annotation_filepaths["predict"],
                auxiliary_dicts=self.auxiliary_dicts,
                input_template=self.input_template,
                output_template=self.output_template,
                label2word=self.label2word,
                task=self.task,
                labels=self.labels
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=8, collate_fn=self.collate_fn)
