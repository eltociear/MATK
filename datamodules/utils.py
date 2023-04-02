import torch
from tqdm import tqdm
import numpy as np
import os
import sys
import csv
import base64
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

def preprocess_image(IMG_PATH, frcnn_cfg, frcnn, image_preprocess):
    
    IMG_ID = None

    # Load features from the original LXMERT repo

    csv.field_size_limit(sys.maxsize)
    FIELDNAMES = [
        "img_id",
        "img_h",
        "img_w",
        "objects_id",
        "objects_conf",
        "attrs_id",
        "attrs_conf",
        "num_boxes",
        "boxes",
        "features",
    ]

    orig_item = None
    if IMG_ID is not None:
        with open(ORIGINAL_REPO_TESTDEV_FEATURES) as f:
            reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
            for item in tqdm(reader):

                for key in ["img_h", "img_w", "num_boxes"]:
                    item[key] = int(item[key])

                boxes = item["num_boxes"]
                decode_config = [
                    ("objects_id", (boxes,), np.int64),
                    ("objects_conf", (boxes,), np.float32),
                    ("attrs_id", (boxes,), np.int64),
                    ("attrs_conf", (boxes,), np.float32),
                    ("boxes", (boxes, 4), np.float32),
                    ("features", (boxes, -1), np.float32),
                ]
                for key, shape, dtype in decode_config:
                    item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                    item[key] = item[key].reshape(shape)
                    item[key].setflags(write=False)

                # Normalize the boxes (to 0 ~ 1)
                item["normalized_boxes"] = item["boxes"].copy()
                img_h, img_w = item["img_h"], item["img_w"]
                item["normalized_boxes"][:, (0, 2)] /= img_w
                item["normalized_boxes"][:, (1, 3)] /= img_h
                item["normalized_boxes"].setflags(write=False)

                if item["img_id"] == IMG_ID:
                    orig_item = item
                    break

    # load models and model components
    
    OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
    ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
    GQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/gqa/trainval_label2ans.json"
    objids = lxmert_utils.get_data(OBJ_URL)
    attrids = lxmert_utils.get_data(ATTR_URL)
    gqa_answers = lxmert_utils.get_data(GQA_URL)
    BOXES_LIMIT=frcnn_cfg.max_detections

    def convert_orig_to_new(item):
        output_dict = {
            "obj_ids": torch.from_numpy(orig_item["objects_id"]).unsqueeze(0),
            "obj_probs": torch.from_numpy(orig_item["objects_conf"]).unsqueeze(0),
            "attr_ids": torch.from_numpy(orig_item["attrs_id"]).unsqueeze(0),
            "attr_probs": torch.from_numpy(orig_item["attrs_conf"]).unsqueeze(0),
            "sizes": torch.Tensor([orig_item["img_h"], orig_item["img_w"]]).unsqueeze(0), # Doesn't match due to resizing
            "preds_per_image": torch.IntTensor([orig_item["num_boxes"]]),
            "roi_features": torch.from_numpy(orig_item["features"]).unsqueeze(0),
            "boxes": torch.from_numpy(orig_item["boxes"]).unsqueeze(0),
            "normalized_boxes": torch.from_numpy(orig_item["normalized_boxes"]).unsqueeze(0)
        }
        return output_dict

    images, sizes, scales_yx = image_preprocess(IMG_PATH)
    output_dict = frcnn(
        images, 
        sizes, 
        scales_yx=scales_yx, 
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt"
    )

    if orig_item is not None:
        frcnn_visualizer = SingleImageViz(IMG_PATH, id2obj=objids, id2attr=attrids)

        x = convert_orig_to_new(orig_item)
        frcnn_visualizer.draw_boxes(
            x.get("boxes")[:,:BOXES_LIMIT],
            x.get("obj_ids")[:,:BOXES_LIMIT],
            x.get("obj_probs")[:,:BOXES_LIMIT],
            x.get("attr_ids")[:,:BOXES_LIMIT],
            x.get("attr_probs")[:,:BOXES_LIMIT],
        )

    return output_dict

def better_preprocess(URL,frcnn_cfg, frcnn, image_preprocess):
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

def image_collate_fn_vlt5(batch, processor, image_processor):
    texts, images = [], []
    for item in batch:
        texts.append(item["text"])
        images.append(item["img_path"])

    ## GETTING INPUT IMAGES FEATURES
    frcnn_cfg = image_processor['frcnn_cfg']
    frcnn = image_processor['frcnn']
    image_preprocess = image_processor['image_preprocess']

    IMG_PATH = "https://images.pexels.com/photos/792381/pexels-photo-792381.jpeg?cs=srgb&dl=pexels-george-desipris-792381.jpg&fm=jpg"

    # output_dict = preprocess_image(IMG_PATH,frcnn_cfg, frcnn, image_preprocess)
    output_dict = better_preprocess(images, frcnn_cfg,frcnn, image_preprocess)
    
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