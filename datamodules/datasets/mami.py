import torch
import numpy as np
import pandas as pd
import lightning.pytorch as pl

from torch.utils.data import DataLoader, Dataset
from PIL import Image

from functools import partial
from torchvision.transforms import ToTensor

from typing import Optional
from transformers import FlavaProcessor, BertTokenizerFast
from .utils import get_collator

from .gqa_lxmert.modeling_frcnn import GeneralizedRCNN
from .gqa_lxmert.lxmert_utils import Config
from .gqa_lxmert.processing_image import Preprocess

from typing import List

class MamiDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, 
                 annotation_filepaths: dict, 
                 img_dir: dict, 
                 model_class_or_path: str, 
                 batch_size: int, 
                 shuffle_train: bool, 
                 labels: List[str],
                 generative_task: bool,
                 features_class_path: str,
                 **kwargs):
        super().__init__()

        self.dataset_class = MamiDataset 
        self.annotation_filepaths = annotation_filepaths
        self.img_dir = img_dir
        self.labels = labels
        self.generative_task = generative_task

        self.batch_size = batch_size
        self.shuffle_train = shuffle_train

        self.batch_size = batch_size
        self.shuffle_train = shuffle_train

        if "flava" in model_class_or_path:
            processor = FlavaProcessor.from_pretrained(model_class_or_path)
            self.collate_fn = partial(image_collate_fn, processor=processor, labels=labels)

        elif ("bert" in model_class_or_path) or ("lxmert" in model_class_or_path):
            processor = BertTokenizerFast.from_pretrained(model_class_or_path)

            if features_class_path is None:
                image_processor = {}
                frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
                frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
                image_preprocess = Preprocess(frcnn_cfg)

                image_processor['frcnn_cfg'] = frcnn_cfg
                image_processor['frcnn'] = frcnn
                image_processor['image_preprocess'] = image_preprocess

                self.collate_fn = partial(image_collate_fn, processor=processor, image_handler=image_processor)
                
            else:
                ## read from the features file 
                with open(features_class_path, 'r') as f:
                    # Load the data from the JSON file into a dictionary
                    read_dict = json.load(f)
                
                output_dict = OrderedDict()
                for key, value in read_dict.items():
                    output_dict[key] = torch.tensor(read_dict[key])
                
                self.collate_fn = partial(image_collate_fn, processor=processor, image_handler=output_dict)

                
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = self.dataset_class(
                annotation_filepath=self.annotation_filepaths["train"],
                img_dir=self.img_dir["train"],
                labels=self.labels,
                generative_task=self.generative_task
            )

            self.validate = self.dataset_class(
                annotation_filepath=self.annotation_filepaths["validate"],
                img_dir=self.img_dir["validate"],
                labels=self.labels,
                generative_task=self.generative_task
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.dataset_class(
                annotation_filepath=self.annotation_filepaths["test"],
                img_dir=self.img_dir["test"],
                labels=self.labels,
                generative_task=self.generative_task
            )

        if stage == "predict" or stage is None:
            self.predict = self.dataset_class(
                annotation_filepath=self.annotation_filepaths["predict"],
                img_dir=self.img_dir["test"],
                labels=self.labels,
                generative_task=self.generative_task
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
    
    def __init__(self, 
                 annotation_filepath: str, 
                 img_dir: str, 
                 labels: List[str], 
                 generative_task: bool):
        self.img_dir = img_dir
        self.labels = labels

        self.img_annotations = pd.read_csv(annotation_filepath, sep='\t')
        if generative_task:
            self.img_annotations = self._transform_labels(self.img_annotations, labels)


    def __len__(self):
        return len(self.img_annotations)

    def _transform_labels(annotations: pd.DataFrame, labels: List[str]):
        for l in labels:
            annotations[l] = annotations[l].apply(lambda x: l if x == 1 else f"not {l}")

        return annotations

    def __getitem__(self, idx):

        file_name = self.img_annotations.loc[idx, 'file_name']
        text = self.img_annotations.loc[idx, "Text Transcription"]

        img_path = self.img_dir + file_name
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img = img.convert("RGB") if img.mode != "RGB" else img

        record = {
            'file_name': file_name,
            'image': np.array(img),
            'text': text,
            'img_path': img_path
        }
        
        for l in self.labels:
            record[l] = self.img_annotations.loc[idx, l]

        return record