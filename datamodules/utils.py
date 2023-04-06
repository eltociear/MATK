import torch
import lightning.pytorch as pl
import torchmetrics
from collections import OrderedDict

from .gqa_lxmert.processing_image import Preprocess
from .gqa_lxmert.visualizing_image import SingleImageViz
import io
from IPython.display import clear_output, Image, display
import PIL.Image
from .gqa_lxmert.modeling_frcnn import GeneralizedRCNN
from .gqa_lxmert.lxmert_utils import Config
from .gqa_lxmert import lxmert_utils

def image_collate_fn(batch, processor):
    texts, images = [], []
    for item in batch:
        texts.append(item["text"])
        images.append(item["image"])
    
    inputs = processor(  
        text=texts, images=images, return_tensors="pt", padding=True
    )

    # Get Labels
    label_name = "label"
    if label_name in batch[0].keys():
        labels = [feature[label_name] for feature in batch]
        inputs['labels'] = torch.tensor(labels, dtype=torch.int64)

    return inputs

def image_collate_fn_mami(batch, processor):
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
        inputs['stereotype'] = torch.tensor(labels, dtype=torch.int64)

    label4 = "objectification"
    if label4 in batch[0].keys():
        labels = [feature[label4] for feature in batch]
        inputs['objectification'] = torch.tensor(labels, dtype=torch.int64)

    label5 = "violence"
    if label5 in batch[0].keys():
        labels = [feature[label5] for feature in batch]
        inputs['violence'] = torch.tensor(labels, dtype=torch.int64)

    return inputs

def image_collate_fn_harmeme(batch, processor):
    texts, images = [], []
    for item in batch:
        texts.append(item["text"])
        images.append(item["image"])
    
    inputs = processor(  
        text=texts, images=images, return_tensors="pt", padding=True
    )

    # Get Labels
    label_name = "labels"

    intensity_map = {'not harmful': 0, 'somewhat harmful': 1, 'very harmful': 2}
    target_map = {'individual': 0, 'organization': 1, 'community': 2 , 'society': 3}

    intensity_labels = []
    target_labels = []

    if label_name in batch[0].keys():
        labels = [feature[label_name] for feature in batch] # [['somewhat harmful', 'community'], ['not harmful']]
        mapped_labels = []
        for one_dim_list in labels:
            for i in range(len(one_dim_list)):
                curr_label = one_dim_list[i]
                
                if curr_label in intensity_map:
                    one_dim_list[i] = intensity_map[curr_label]
                else:
                    one_dim_list[i] = target_map[curr_label]
            
            mapped_labels.append(one_dim_list)
        
        for l in mapped_labels:
            if len(l) == 2:
                intensity_labels.append(l[0])
                target_labels.append(l[1])
            else:
                intensity_labels.append(l[0])
                target_labels.append(0)
        
        intensity_tensor = torch.tensor(intensity_labels, dtype=torch.int64)
        target_tensor = torch.tensor(target_labels, dtype=torch.int64)

        inputs['intensity'] = intensity_tensor
        inputs['target'] = target_tensor
        
    return inputs

def preprocess_image(URL,frcnn_cfg, frcnn, image_preprocess):
    # run frcnn
    images, sizes, scales_yx = image_preprocess(URL)
    output_dict = frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt",
    )
    return output_dict

def image_collate_fn_fhm_visualbert(batch, processor, image_handler):
    texts, images = [], []
    for item in batch:
        texts.append(item["text"])
        images.append(item["img_path"])


    if isinstance(image_handler, OrderedDict):
        output_dict = image_handler

    else:
        ## GETTING INPUT IMAGES FEATURES
        frcnn_cfg = image_handler['frcnn_cfg']
        frcnn = image_handler['frcnn']
        image_preprocess = image_handler['image_preprocess']

        output_dict = preprocess_image(images, frcnn_cfg,frcnn, image_preprocess)
        
        # writing_dict = {}
        # for key, value in output_dict.items():
        #    x_numpy = output_dict[key].numpy()
        #    x_list = x_numpy.tolist()
        #    writing_dict[key] = x_list
           
        # with open('data.json', 'w') as f:
        #     # Write the dictionary to the JSON file
        #     json.dump(writing_dict, f)
    
    inputs = processor(
            texts,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

    # Get Labels
    label_name = "label"
    if label_name in batch[0].keys():
        labels = [feature[label_name] for feature in batch]
        inputs['labels'] = torch.tensor(labels, dtype=torch.int64)
    inputs['visual_feats'] = output_dict.get("roi_features")
    inputs['visual_pos'] = visual_pos=output_dict.get("normalized_boxes")
    return inputs

def image_collate_fn_mami_visualbert(batch, processor, image_handler):
    texts, images = [], []
    for item in batch:
        texts.append(item["text"])
        images.append(item["img_path"])

    if isinstance(image_handler, OrderedDict):
        output_dict = image_handler

    else:
        ## GETTING INPUT IMAGES FEATURES
        frcnn_cfg = image_handler['frcnn_cfg']
        frcnn = image_handler['frcnn']
        image_preprocess = image_handler['image_preprocess']

        output_dict = preprocess_image(images, frcnn_cfg,frcnn, image_preprocess)
        
        # writing_dict = {}
        # for key, value in output_dict.items():
        #    x_numpy = output_dict[key].numpy()
        #    x_list = x_numpy.tolist()
        #    writing_dict[key] = x_list
           
        # with open('data.json', 'w') as f:
        #     # Write the dictionary to the JSON file
        #     json.dump(writing_dict, f)
    
    inputs = processor(
            texts,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
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
        inputs['stereotype'] = torch.tensor(labels, dtype=torch.int64)

    label4 = "objectification"
    if label4 in batch[0].keys():
        labels = [feature[label4] for feature in batch]
        inputs['objectification'] = torch.tensor(labels, dtype=torch.int64)

    label5 = "violence"
    if label5 in batch[0].keys():
        labels = [feature[label5] for feature in batch]
        inputs['violence'] = torch.tensor(labels, dtype=torch.int64)

    inputs['visual_feats'] = output_dict.get("roi_features")
    inputs['visual_pos'] = visual_pos=output_dict.get("normalized_boxes")
    return inputs

def image_collate_fn_harmeme_visualbert(batch, processor, image_handler):
    texts, images = [], []
    for item in batch:
        texts.append(item["text"])
        images.append(item["img_path"])
    
    if isinstance(image_handler, OrderedDict):
        output_dict = image_handler

    else:
        ## GETTING INPUT IMAGES FEATURES
        frcnn_cfg = image_handler['frcnn_cfg']
        frcnn = image_handler['frcnn']
        image_preprocess = image_handler['image_preprocess']

        output_dict = preprocess_image(images, frcnn_cfg,frcnn, image_preprocess)
        
    inputs = processor(
            texts,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

    # Get Labels
    label_name = "labels"

    intensity_map = {'not harmful': 0, 'somewhat harmful': 1, 'very harmful': 2}
    target_map = {'individual': 0, 'organization': 1, 'community': 2 , 'society': 3}

    intensity_labels = []
    target_labels = []

    if label_name in batch[0].keys():
        labels = [feature[label_name] for feature in batch] # [['somewhat harmful', 'community'], ['not harmful']]
        mapped_labels = []
        for one_dim_list in labels:
            for i in range(len(one_dim_list)):
                curr_label = one_dim_list[i]
                
                if curr_label in intensity_map:
                    one_dim_list[i] = intensity_map[curr_label]
                else:
                    one_dim_list[i] = target_map[curr_label]
            
            mapped_labels.append(one_dim_list)
        
        for l in mapped_labels:
            if len(l) == 2:
                intensity_labels.append(l[0])
                target_labels.append(l[1])
            else:
                intensity_labels.append(l[0])
                target_labels.append(0)
        
        intensity_tensor = torch.tensor(intensity_labels, dtype=torch.int64)
        target_tensor = torch.tensor(target_labels, dtype=torch.int64)

        inputs['intensity'] = intensity_tensor
        inputs['target'] = target_tensor

    inputs['visual_feats'] = output_dict.get("roi_features")
    inputs['visual_pos'] = visual_pos=output_dict.get("normalized_boxes")

    return inputs