import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from PIL import Image

from functools import partial
from torchvision.transforms import ToTensor

from typing import Optional
from transformers import FlavaProcessor


def image_collate_fn(batch, processor):
    texts, images = [], []
    for item in batch:
        texts.append(item["text"])
        images.append(item["image"])
    
    inputs = processor(  
        text=texts, images=images, return_tensors="pt", padding=True
    )

    # Get Labels
    label1 = "misogynous"
    if label1 in batch[0].keys():
        labels = [feature[label1] for feature in batch]
        inputs['misogynous'] = torch.tensor(labels, dtype=torch.int64)

    label2 = "shaming"
    if label2 in batch[0].keys():
        labels = [feature[label2] for feature in batch]
        inputs["shaming"] = torch.tensor(labels, dtype=torch.int64)

    label3 = "stereotype"
    if label3 in batch[0].keys():
        labels = [feature[label3] for feature in batch]
        inputs['misogynous'] = torch.tensor(labels, dtype=torch.int64)

    label4 = "objectification"
    if label4 in batch[0].keys():
        labels = [feature[label4] for feature in batch]
        inputs['objectification'] = torch.tensor(labels, dtype=torch.int64)

    label5 = "violence"
    if label5 in batch[0].keys():
        labels = [feature[label5] for feature in batch]
        inputs['violence'] = torch.tensor(labels, dtype=torch.int64)

    return inputs

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