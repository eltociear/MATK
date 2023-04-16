
visual_features_file_path = "hf_generated.datasets_testdev.datasets"
# visual_features_file_path = "vg_gqa_imgfeat/gqa_testdev_obj36.tsv"

import os
import json
import datasets
import numpy as np
from transformers import LxmertForQuestionAnswering, LxmertTokenizer
import .lxmert_utils
import torch
from tqdm import tqdm

GQA_QUESTIONS_DIR = "."
GQA_QUESTIONS_FILE = os.path.join(GQA_QUESTIONS_DIR, "testdev_balanced_questions.json")

with open(GQA_QUESTIONS_FILE) as dataset_file:
    dataset = json.load(dataset_file)

imgid2img = {}
if visual_features_file_path.endswith(".datasets"):
    visual_features = datasets.Dataset.from_file(visual_features_file_path)

    decode_config = [
        ("preds_per_image", "num_boxes", np.int64),
        ("obj_ids", "objects_id", np.int64),
        ("obj_probs", "objects_conf", np.float32),
        ("attr_ids", "attrs_id", np.int64),
        ("attr_probs", "attrs_conf", np.float32),
        ("boxes", "boxes", np.float32),
        ("normalized_boxes", "normalized_boxes", np.float32),
        ("roi_features", "features", np.float32),
    ]

    for full_item in visual_features:
        item = {}
        for key, new_key, dtype in decode_config:
            item[new_key] = np.array(full_item[key], dtype=dtype)
            item[new_key].setflags(write=False)

        item["img_h"] = int(full_item["sizes"][0])
        item["img_w"] = int(full_item["sizes"][1])

        imgid2img[full_item["img_id"]] = item
else:
    import sys
    import csv
    import base64

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
    with open(visual_features_file_path) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for item in reader:

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

            imgid2img[item["img_id"]] = item

lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
lxmert_gqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-gqa-uncased").to("cuda")

GQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/gqa/trainval_label2ans.json"
gqa_answers = lxmert_utils.get_data(GQA_URL)

accuracy = 0
for question_id, question_obj in tqdm(dataset.items()):
    inputs = lxmert_tokenizer(
        question_obj["question"],
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    ).to("cuda")

    output_gqa = lxmert_gqa(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        visual_feats=torch.from_numpy(imgid2img[question_obj["imageId"]].get("features")).unsqueeze(0).to("cuda"),
        visual_pos=torch.from_numpy(imgid2img[question_obj["imageId"]].get("normalized_boxes")).unsqueeze(0).to("cuda"),
        token_type_ids=inputs.token_type_ids,
        output_attentions=False,
    )

    pred_gqa = output_gqa["question_answering_score"].argmax(-1)
    accuracy += int(question_obj["answer"] == gqa_answers[pred_gqa])
accuracy = accuracy / len(dataset)

print(f"Accuracy: {accuracy}")
