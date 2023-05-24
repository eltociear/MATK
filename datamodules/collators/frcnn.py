import torch

def preprocess_image(
        urls, 
        image_preprocess, 
        frcnn, 
        frcnn_cfg
    ):
    # run frcnn
    images, sizes, scales_yx = image_preprocess(urls)
    output_dict = frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt",
    )
    return output_dict

def image_collate_fn(
        batch, 
        tokenizer,
        frcnn,
        frcnn_cfg,
        image_preprocess,
        labels
    ):
    images = []
    for item in batch:
        images.append(item["img_path"])

    output_dict = preprocess_image(images, image_preprocess, frcnn, frcnn_cfg)
    
    return _image_collate_fn(
        batch=batch,
        tokenizer=tokenizer,
        labels=labels,
        output_dict=output_dict
    )

def image_collate_fn_fast(
        batch, 
        tokenizer,
        feats_dict,
        labels
    ):
    return _image_collate_fn(
        batch=batch,
        tokenizer=tokenizer,
        labels=labels,
        output_dict=feats_dict
    )

def _image_collate_fn(
        batch, 
        tokenizer,
        labels,
        output_dict
    ):
    texts = []
    for item in batch:
        texts.append(item["text"])
    
    inputs = tokenizer(
            texts,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

    inputs['visual_feats'] = output_dict.get("roi_features")
    inputs['visual_pos'] = output_dict.get("normalized_boxes")

    # Get Labels
    for l in labels:
        if l in batch[0].keys():
            labels = [feature[l] for feature in batch]
            inputs[l] = torch.tensor(labels, dtype=torch.int64)
            
    return inputs