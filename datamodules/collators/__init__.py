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
            
        if "feats_dict" in kwargs:
            return partial(
                frcnn_collator_fast, 
                tokenizer=tokenizer, 
                labels=labels, 
                feats_dict=kwargs["feats_dict"],
                feats_info_dict=kwargs["feats_info_dict"]
            )
        else:
            frcnn_class_or_path = kwargs.pop("frcnn_class_or_path")
            frcnn_cfg = Config.from_pretrained(frcnn_class_or_path)
            image_preprocess = Preprocess(frcnn_cfg)

            return partial(
                frcnn_collator, 
                tokenizer=tokenizer, 
                labels=labels,
                image_preprocess=image_preprocess
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_class_or_path)
        return partial(text_collate_fn, tokenizer=tokenizer, labels=labels)