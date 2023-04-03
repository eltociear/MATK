import torch
import lightning.pytorch as pl
import torchmetrics

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