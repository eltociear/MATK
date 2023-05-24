import json
import torch

from functools import partial
from collections import OrderedDict
from transformers import FlavaProcessor, AutoTokenizer

from .gqa_lxmert.modeling_frcnn import GeneralizedRCNN
from .gqa_lxmert.lxmert_utils import Config
from .gqa_lxmert.processing_image import Preprocess

# Flava
from .flava import image_collate_fn as flava_collator

# LXMERT, VisualBERT
from .frcnn import image_collate_fn as frcnn_collator
from .frcnn import image_collate_fn_fast as frcnn_collator_fast

from .text import text_collate_fn

def get_collator(
    model_class_or_path,
    labels,
    **kwargs
):
    bert_tokenizer_models = ["lxmert", "visualbert"]

    if "flava" in model_class_or_path:
        processor = FlavaProcessor.from_pretrained(model_class_or_path)
        return partial(flava_collator, processor=processor, labels=labels)
    elif any([x in model_class_or_path for x in bert_tokenizer_models]):
        tokenizer = AutoTokenizer.from_pretrained(model_class_or_path)

        if "feats_filepath" not in kwargs or kwargs["feats_filepath"] is None:
            frcnn_model_or_path = kwargs.pop("frcnn_model_or_path")

            frcnn_cfg = Config.from_pretrained(frcnn_model_or_path)
            frcnn = GeneralizedRCNN.from_pretrained(frcnn_model_or_path, config=frcnn_cfg)
            image_preprocess = Preprocess(frcnn_cfg)

            return partial(
                frcnn_collator_fast, 
                tokenizer=tokenizer, 
                frcnn_cfg=frcnn_cfg, 
                frcnn=frcnn, 
                image_preprocess=image_preprocess,
                labels=labels
            )
            
        else:
            ## read from the features file 
            with open(kwargs["feats_filepath"], 'r') as f:
                # Load the data from the JSON file into a dictionary
                read_dict = json.load(f)
            
            feats_dict = OrderedDict()
            for k, v in read_dict.items():
                feats_dict[k] = torch.tensor(v)
            
            return partial(frcnn_collator, tokenizer=tokenizer, feats_dict=feats_dict)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_class_or_path)
        return partial(text_collate_fn, tokenizer=tokenizer, labels=labels)