import torch

def _image_collate_fn(
        batch, 
        tokenizer,
        labels
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

    # Get Labels
    for l in labels:
        if l in batch[0].keys():
            labels = [feature[l] for feature in batch]
            inputs[l] = torch.tensor(labels, dtype=torch.int64)
            
    return inputs

def image_collate_fn(
        batch, 
        tokenizer,
        image_preprocess,
        labels
    ):
    
    inputs = _image_collate_fn(
        batch=batch,
        tokenizer=tokenizer,
        labels=labels
    )

    images = []
    for item in batch:
        images.append(item["image_path"])

    images, sizes, scales_yx = image_preprocess(images)

    inputs["images"] = images
    inputs["sizes"] = sizes
    inputs["scales_yx"] = scales_yx
    
    return inputs

def image_collate_fn_fast(
        batch, 
        tokenizer,
        feats_dict,
        feats_info_dict,
        labels
    ):
    inputs = _image_collate_fn(
        batch=batch,
        tokenizer=tokenizer,
        labels=labels
    )


    visual_feats, visual_pos = [], []
    for item in batch:
        item_id = str(item["id"])
        visual_feats.append(feats_dict[item_id])
        visual_pos.append(feats_info_dict[f"{item_id}_info"]["normalized_boxes"])

    inputs['visual_feats'] = visual_feats
    inputs['visual_pos'] = visual_pos

    return inputs