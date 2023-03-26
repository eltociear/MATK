import torch

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