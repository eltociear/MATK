import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics


from transformers import FlavaModel

class MetricCallback(pl.Callback):
    def __init__(self, metric_dict):
        super().__init__()
        self.metric_dict = metric_dict
        
    def on_train_start(self, trainer, pl_module):
        for key, value in self.metric_dict.items():
            print(pl_module)
            setattr(pl_module, f"{key}_train_acc", torchmetrics.Accuracy(task="multiclass", num_classes=value).to('cuda'))
            setattr(pl_module, f"{key}_train_auroc", torchmetrics.AUROC(task="multiclass", num_classes=value).to('cuda'))
            
    def on_validation_start(self, trainer, pl_module):
        for key, value in self.metric_dict.items():
            setattr(pl_module, f"{key}_val_acc", torchmetrics.Accuracy(task="multiclass", num_classes=value).to('cuda'))
            setattr(pl_module, f"{key}_val_auroc", torchmetrics.AUROC(task="multiclass", num_classes=value).to('cuda'))
            
    def on_test_start(self, trainer, pl_module):
        for key, value in self.metric_dict.items():
            setattr(pl_module, f"{key}_test_acc", torchmetrics.Accuracy(task="multiclass", num_classes=value).to('cuda'))
            setattr(pl_module, f"{key}_test_auroc", torchmetrics.AUROC(task="multiclass", num_classes=value).to('cuda'))


class FlavaClassificationModel(pl.LightningModule):
    def __init__(self, model_class_or_path, output_dict):
        super().__init__()
        self.save_hyperparameters()
        self.model = FlavaModel.from_pretrained(model_class_or_path)

        # set up classification
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(self.model.config.multimodal_config.hidden_size, value)) for key, value in output_dict.items()])
        
        # set up metric
        self.metric_dict = output_dict
       

    def compute_metrics_and_logs(self, preds, labels, loss, prefix, step):
        acc = getattr(self, f"{prefix}_{step}_acc")
        auroc = getattr(self, f"{prefix}_{step}_auroc")

        acc(preds.argmax(dim=-1), labels)
        auroc(preds, labels)

        self.log(f'{prefix}_{step}_loss', loss, prog_bar=True)
        self.log(f'{prefix}_{step}_acc', acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}_{step}_auroc', auroc, on_step=True, on_epoch=True, sync_dist=True)


    def training_step(self, batch, batch_idx):

        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )

        loss = 0

        label_list = []
        for k,v in self.metric_dict.items():
            label_list.append(k)

        for i in range(len(self.metric_dict)):
            label_targets = batch[label_list[i]]
            label_preds = self.mlps[i](model_outputs.multimodal_embeddings[:, 0])
            label_loss = F.cross_entropy(label_preds, label_targets)
            loss += label_loss
            self.compute_metrics_and_logs(label_preds, label_targets, label_loss,label_list[i] , 'train')
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )

        loss = 0

        label_list = []
        for k,v in self.metric_dict.items():
            label_list.append(k)


        for i in range(len(self.metric_dict)):
            label_targets = batch[label_list[i]]
            label_preds = self.mlps[i](model_outputs.multimodal_embeddings[:, 0])
            label_loss = F.cross_entropy(label_preds, label_targets)
            loss += label_loss
            self.compute_metrics_and_logs(label_preds, label_targets, label_loss,label_list[i] , 'val')
        
        return loss


    def test_step(self, batch, batch_idx):

        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )

        loss = 0

        label_list = []
        for k,v in self.metric_dict.items():
            label_list.append(k)


        for i in range(len(self.metric_dict)):
            label_targets = batch[label_list[i]]
            label_preds = self.mlps[i](model_outputs.multimodal_embeddings[:, 0])
            label_loss = F.cross_entropy(label_preds, label_targets)
            loss += label_loss
            self.compute_metrics_and_logs(label_preds, label_targets, label_loss,label_list[i] , 'test')

        return None

    def on_test_epoch_end(self):

        print("shaming_test_acc:", self.shaming_test_acc.compute())
        print("shaming_test_auroc:", self.shaming_test_auroc.compute())
        print("misogynous_test_acc:", self.misogynous_test_acc.compute())
        print("misogynous_test_auroc:", self.misogynous_test_auroc.compute())
        print("stereotype_test_acc:", self.stereotype_test_acc.compute())
        print("stereotype_test_auroc:", self.stereotype_test_auroc.compute())
        print("objectification_test_acc:", self.objectification_test_acc.compute())
        print("objectification_test_auroc:", self.objectification_test_auroc.compute())
        print("objectification_test_acc:", self.objectification_test_acc.compute())
        print("objectification_test_auroc:", self.objectification_test_auroc.compute())
        

    def predict_step(self, batch, batch_idx):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )

        loss = 0

        label_list = []
        for k,v in self.metric_dict.items():
            label_list.append(k)

        results = {}
        for i in range(len(self.metric_dict)):
            label_targets = batch[label_list[i]]
            label_preds = self.mlps[i](model_outputs.multimodal_embeddings[:, 0])
            results[label_list[i]] = label_preds

        if "labels" in batch:
            results['labels'] = batch["labels"]

        return results
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return [self.optimizer]