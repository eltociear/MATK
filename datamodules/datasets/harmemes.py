import os
import tqdm
import numpy as np
import pickle as pkl

from PIL import Image

from datamodules.datasets.prompthatebase import MultimodalDataBase
from . import utils
import torch
import random
from tqdm import tqdm
from typing import List
from torch.utils.data import Dataset


INTENSITY_MAP = {
    'not harmful': 0, 
    'somewhat harmful': 1, 
    'very harmful': 2
}

TARGET_MAP = {
    'individual': 0, 
    'organization': 1, 
    'community': 2 , 
    'society': 3
}

class HarmemesBase(Dataset):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        labels: List[str]
    ):
        self.annotations = self._preprocess_annotations(annotation_filepath)
        self.auxiliary_data = self._load_auxiliary(auxiliary_dicts)
        self.labels = labels

    def _preprocess_annotations(self, annotation_filepath: str):
        annotations = []

        # load the default annotations
        data = utils._load_jsonl(annotation_filepath)

        record_id = 0
        
        # translate labels into numeric values
        for record in tqdm.tqdm(data, desc="Preprocessing labels"):
            record["img"] = record.pop("image")
            record["intensity"] = INTENSITY_MAP[record["labels"][0]]
            record["target"] = TARGET_MAP[record["labels"][1]] if len(record["labels"]) > 1 else -1
            record["id"] = record_id
            record_id += 1
            annotations.append(record)
        
        return annotations

    def _load_auxiliary(self, auxiliary_dicts: dict):
        data = {}
        for key, filepath in tqdm.tqdm(auxiliary_dicts.items(), desc="Loading auxiliary info"):
            with open(filepath, "rb") as f:
                data[key] = pkl.load(f)

        return data

    def __len__(self):
        return len(self.annotations)


class FasterRCNNDataset(HarmemesBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        labels: List[str],
        feats_dict: dict
    ):
        super().__init__(annotation_filepath, auxiliary_dicts, labels)
        self.feats_dict = feats_dict

    def __getitem__(self, idx: int):
        record = self.annotations[idx]

        text = record['text']
        image_id = record['img']
        id, _ = os.path.splitext(image_id)

        item = {
            'id': id,
            'image_id': image_id,
            'text': text,
            'roi_features': self.feats_dict[id]['roi_features'],
            'normalized_boxes': self.feats_dict[id]['normalized_boxes']
        }

        for l in self.labels:
            item[l] = record[l]

        return item


class ImagesDataset(HarmemesBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        labels: List[str],
        image_dir: str
    ):
        super().__init__(annotation_filepath, auxiliary_dicts, labels)
        self.image_dir = image_dir

    def __getitem__(self, idx: int):
        record = self.annotations[idx]

        image_filename = record['img']
        image_id, _ = os.path.splitext(image_filename)

        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = image.convert("RGB") if image.mode != "RGB" else image

        item = {
            'id': record['id'],
            'image_id': image_id,
            'text': record['text'],
            'image': np.array(image),
            'image_path': image_path
        }

        for l in self.labels:
            item[l] = record[l]

        return item


class TextDataset(HarmemesBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        labels: List[str],
        input_template: str,
        output_template: str,
        label2word: dict
    ):
        super().__init__(annotation_filepath, auxiliary_dicts, labels)
        self.input_template = input_template
        self.output_template = output_template
        self.label2word = label2word

    def __getitem__(self, idx: int):
        record = self.annotations[idx]

        # Format the input template
        input_kwargs = {"text": record['text']}
        for key, data in self.auxiliary_data.items():
            input_kwargs[key] = data[f"{id:05}"]

        image_id, _ = os.path.splitext(record['img'])

        item = {
            'id': record["id"],
            'image_id': image_id,
            'text': self.input_template.format(**input_kwargs)
        }

        for l in self.labels:
            label = record[l]
            item[l] = self.output_template.format(label=self.label2word[label])

        return item

class MultimodalDataset(MultimodalDataBase):
    #mem, off, harm
    def __init__(self,opt,tokenizer,dataset,path, cap_path, prompt_arch,mode='train',few_shot_index=0):
        
        super().__init__(opt,tokenizer,dataset,path, cap_path, prompt_arch, mode='train',few_shot_index=0)

        if self.fine_grind:
            self.label_mapping_word={0:'nobody',
                                     1:'society',
                                     2:'individual',
                                     3:'community',
                                     4:'organization'}
            self.attack_list={'society':0,
                              'individual':1,
                              'community':2,
                              'organization':3}
            self.attack_file=utils.load_pkl(os.path.join(self.opt['DATA'],
                                                       'domain_splits','harm_trgt.pkl'))
            self.template="*<s>**sent_0*.*_It_was_targeting*label_**</s>*"

        else:
            self.label_mapping_word={0:self.opt['POS_WORD'],
                                 1:self.opt['NEG_WORD']}
            self.template="*<s>**sent_0*.*_It_was*label_**</s>*"
            
        self.label_mapping_id={}
        for label in self.label_mapping_word.keys():
            mapping_word=self.label_mapping_word[label]
            #add space already
            assert len(tokenizer.tokenize(' ' + self.label_mapping_word[label])) == 1
            self.label_mapping_id[label] = \
            tokenizer._convert_token_to_id(
                tokenizer.tokenize(' ' + self.label_mapping_word[label])[0])
            print ('Mapping for label %d, word %s, index %d' % 
                   (label,mapping_word,self.label_mapping_id[label]))
        #implementation for one template now

        self.template_list=self.template.split('*')
        self.special_token_mapping = {
            '<s>': tokenizer.convert_tokens_to_ids('<s>'),
            '<mask>': tokenizer.mask_token_id, 
            '<pad>': tokenizer.pad_token_id, #1 for roberta
            '</s>': tokenizer.convert_tokens_to_ids('<\s>') 
        }
        
        self.support_examples=self.load_entries(path, cap_path)
        
        self.entries=self.load_entries(path, cap_path)
        if self.opt['DEBUG']:
            self.entries=self.entries[:128]
        self.prepare_exp()
        

    def load_entries(self,path, cap_path):
        
        data=utils.read_json(path)
        captions=utils.load_pkl(cap_path)
        
        entries=[]
        for k,row in enumerate(data):
            label=row['label']
            img=row['img']
            cap=captions[img.split('.')[0]][:-1]#remove the punctuation in the end
            sent=row['clean_sent']
            #remember the punctuations at the end of each sentence
            cap=cap+' . '+sent+' . '
            #whether using external knowledge
            if self.add_ent:
                cap=cap+' . '+row['entity']+' . '
            if self.add_dem:
                cap=cap+' . '+row['race']+' . '
            entry={
                'cap':cap.strip(),
                'label':label,
                'img':img
            }
            if self.fine_grind:
                if label==0:
                    #[1,0,0,0,0,0]
                    entry['attack']=[1,0,0,0,0]
                else:
                    attack=[0,0,0,0,0]
                    attack_idx=self.attack_list[self.attack_file[img]]+1
                    attack[attack_idx]=1
                    entry['attack']=attack
            
            entries.append(entry)
        return entries
       
    def __getitem__(self,index):
        batch, entry = self.support_getitem(index)
        if self.fine_grind:
            batch['attack']=torch.Tensor(entry['attack'])
        return batch