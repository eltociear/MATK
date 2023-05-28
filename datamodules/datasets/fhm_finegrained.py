import os
import numpy as np

from PIL import Image
from .base import LanguageBase, VisionLanguageBase

from typing import List

class VLFeaturesDataset(VisionLanguageBase):
    def __init__(
        self,
        annotation_filepath: str,
        task: str,
        labels: List[str]
    ):
        super().__init__(annotation_filepath, None, task, labels)

    def __getitem__(self, idx: int):
        if self.task == "hateful_cls":
            return self.get_hateful_cls(idx)
        else:
            raise NotImplementedError(f"'get_{self.task}' is not defined")

    def get_hateful_cls(self, idx):
        id = self.annotations.loc[idx, 'id']
        image_id = self.annotations.loc[idx, 'img']
        text = self.annotations.loc[idx, 'text']

        item = {
            'id': id,
            'image_id': image_id,
            'text': text,
        }

        for l in self.labels:
            item[l] = self.annotations.loc[idx, l]

        return item

class VLImagesDataset(VisionLanguageBase):
    def __init__(
        self,
        annotation_filepath: str,
        image_dir: str,
        task: str,
        labels: List[str]
    ):
        super().__init__(annotation_filepath, image_dir, task, labels)

    def __getitem__(self, idx: int):
        if self.task == "hateful_cls":
            return self.get_hateful_cls(idx)
        else:
            raise NotImplementedError(f"'get_{self.task}' is not defined")

    def get_hateful_cls(self, idx):
        id = self.annotations.loc[idx, 'id']
        image_id = self.annotations.loc[idx, 'img']
        text = self.annotations.loc[idx, 'text']
        label = self.annotations.loc[idx, 'hate']

        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = image.convert("RGB") if image.mode != "RGB" else image

        return {
            'id': id,
            'image_id': image_id,
            'text': text,
            'image': np.array(image),
            'image_path': image_path,
            'label': label
        }


class LanguageDataset(LanguageBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        input_template: str,
        output_template: str,
        label2word: dict,
        task: str,
        labels: List[str]
    ):
        super().__init__(annotation_filepath, auxiliary_dicts,
                         input_template, output_template, label2word, 
                         task, labels)

    def __getitem__(self, idx: int):
        if self.task == "hateful_cls":
            return self.get_hateful_cls(idx)
        else:
            raise NotImplementedError(f"'get_{self.task}' is not defined")

    def get_hateful_cls(self, idx):
        id = self.annotations.loc[idx, 'id']
        image_id = self.annotations.loc[idx, 'img']
        text = self.annotations.loc[idx, 'text']
        label = self.annotations.loc[idx, 'hate']

        # Format the input template
        input_kwargs = {"text": text}
        for key, data in self.auxiliary_data.items():
            input_kwargs[key] = data[f"{id:05}"]

        return {
            'id': id,
            'image_id': image_id,
            'text': self.input_template.format(**input_kwargs),
            'label': self.output_template.format(label=self.label2word[label])
        }
