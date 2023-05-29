import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import torchmetrics

from transformers import FlavaModel

class MetricCallback(Callback):
    def __init__(self, output_dict):
        super().__init__()
        self.output_dict = output_dict
        
    def on_train_start(self, trainer, pl_module):
        for key, value in self.output_dict.items():
            setattr(pl_module, f"{key}_train_acc", torchmetrics.Accuracy(task="multiclass", num_classes=value).to('cuda'))
            setattr(pl_module, f"{key}_train_auroc", torchmetrics.AUROC(task="multiclass", num_classes=value).to('cuda'))
            
    def on_validation_start(self, trainer, pl_module):
        for key, value in self.output_dict.items():
            setattr(pl_module, f"{key}_val_acc", torchmetrics.Accuracy(task="multiclass", num_classes=value).to('cuda'))
            setattr(pl_module, f"{key}_val_auroc", torchmetrics.AUROC(task="multiclass", num_classes=value).to('cuda'))
            
    def on_test_start(self, trainer, pl_module):
        for key, value in self.output_dict.items():
            setattr(pl_module, f"{key}_test_acc", torchmetrics.Accuracy(task="multiclass", num_classes=value).to('cuda'))
            setattr(pl_module, f"{key}_test_auroc", torchmetrics.AUROC(task="multiclass", num_classes=value).to('cuda'))


class FlavaClassificationModel(pl.LightningModule):
    def __init__(
            self, 
            model_class_or_path: str,
            cls_dict: dict
        ):
        super().__init__()
        self.save_hyperparameters()

        self.model = FlavaModel.from_pretrained(model_class_or_path)

        # set up classification
        self.mlps = nn.ModuleList([
            nn.Linear(self.model.config.multimodal_config.hidden_size, value) 
            for value in cls_dict.values()
        ])
        
        # set up metric
        self.cls_dict = cls_dict
        self.metric_callback = MetricCallback(self.cls_dict)
       

    def compute_metrics_and_logs(self, preds, labels, loss, prefix, step):
        acc = getattr(self, f"{prefix}_{step}_acc")
        auroc = getattr(self, f"{prefix}_{step}_auroc")

        acc_value = acc(preds.argmax(dim=-1), labels)
        auroc_value = auroc(preds, labels)

        self.log(f'{prefix}_{step}_loss', loss, prog_bar=True)
        self.log(f'{prefix}_{step}_acc', acc_value, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}_{step}_auroc', auroc_value, on_step=True, on_epoch=True, sync_dist=True)


    def training_step(self, batch, batch_idx):

        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )

        loss = 0

        label_list = []
        for k,v in self.cls_dict.items():
            label_list.append(k)

        for i in range(len(self.cls_dict)):
            label_targets = batch[label_list[i]]
            label_preds = self.mlps[i](model_outputs.multimodal_embeddings[:, 0])
            label_loss = F.cross_entropy(label_preds, label_targets)
            loss += label_loss
            # self.compute_metrics_and_logs(label_preds, label_targets, label_loss,label_list[i] , 'train')
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )

        loss = 0

        label_list = []
        for k,v in self.cls_dict.items():
            label_list.append(k)

        for i in range(len(self.cls_dict)):
            label_targets = batch[label_list[i]]
            label_preds = self.mlps[i](model_outputs.multimodal_embeddings[:, 0])
            label_loss = F.cross_entropy(label_preds, label_targets)
            loss += label_loss
            
            # self.compute_metrics_and_logs(label_preds, label_targets, label_loss,label_list[i] , 'val')
        
        return loss

    def predict_step(self, batch, batch_idx):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )

        loss = 0

        label_list = []
        for k,v in self.cls_dict.items():
            label_list.append(k)

        results = {}
        for i in range(len(self.cls_dict)):
            label_targets = batch[label_list[i]]
            label_preds = self.mlps[i](model_outputs.multimodal_embeddings[:, 0])
            results[label_list[i]] = label_preds

        if "labels" in batch:
            results['labels'] = batch["labels"]

        return results
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return [self.optimizer]