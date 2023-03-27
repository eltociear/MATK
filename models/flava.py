import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics


from transformers import FlavaModel

class FlavaClassificationModel(pl.LightningModule):
    def __init__(self, model_class_or_path, intensity_classes=3, target_classes = 4):
        super().__init__()
        self.save_hyperparameters()
        self.model = FlavaModel.from_pretrained(model_class_or_path)
        self.mlp = nn.Sequential(
            nn.Linear(self.model.config.multimodal_config.hidden_size, intensity_classes)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(self.model.config.multimodal_config.hidden_size, target_classes)
        )

        self.intensity_train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=intensity_classes)
        self.intensity_train_auroc = torchmetrics.AUROC(task="multiclass", num_classes=intensity_classes)

        self.intensity_val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=intensity_classes)
        self.intensity_val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=intensity_classes)

        self.intensity_test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=intensity_classes)
        self.intensity_test_auroc = torchmetrics.AUROC(task="multiclass", num_classes=intensity_classes)

        self.target_train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=target_classes)
        self.target_train_auroc = torchmetrics.AUROC(task="multiclass", num_classes=target_classes)

        self.target_val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=target_classes)
        self.target_val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=target_classes)

        self.target_test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=target_classes)
        self.target_test_auroc = torchmetrics.AUROC(task="multiclass", num_classes=target_classes)

        
    
    def training_step(self, batch, batch_idx):
        intensity_labels = batch['intensity_labels']
        target_labels = batch['target_labels']
        
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )
        
        intensity_preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        target_preds = self.mlp2(model_outputs.multimodal_embeddings[:, 0])

        intensity_loss = F.cross_entropy(intensity_preds, intensity_labels) 
        target_loss = F.cross_entropy(target_preds, target_labels) 
        
        self.intensity_train_acc(intensity_preds.argmax(dim=-1), intensity_labels)
        self.intensity_train_auroc(intensity_preds, intensity_labels)

        self.log('intensity_train_loss', intensity_loss, prog_bar=True)
        self.log('intensity_train_acc', self.intensity_train_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('intensity_train_auroc', self.intensity_train_auroc, on_step=True, on_epoch=True, sync_dist=True)

        self.target_train_acc(target_preds.argmax(dim=-1), target_labels)
        self.target_train_auroc(target_preds, target_labels)

        self.log('target_train_loss', target_loss, prog_bar=True)
        self.log('target_train_acc', self.target_train_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('target_train_auroc', self.target_train_auroc, on_step=True, on_epoch=True, sync_dist=True)

        loss = intensity_loss+target_loss 
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        intensity_labels = batch['intensity_labels']
        target_labels = batch['target_labels']


        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )

        intensity_preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        target_preds = self.mlp2(model_outputs.multimodal_embeddings[:, 0])

        intensity_loss = F.cross_entropy(intensity_preds, intensity_labels) 
        target_loss = F.cross_entropy(target_preds, target_labels) 
        
        self.intensity_val_acc(intensity_preds.argmax(dim=-1), intensity_labels)
        self.intensity_val_auroc(intensity_preds, intensity_labels)

        self.log('intensity_val_loss', intensity_loss, prog_bar=True, sync_dist=True)
        self.log('intensity_val_acc', self.intensity_val_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('intensity_val_auroc', self.intensity_val_auroc, on_step=True, on_epoch=True, sync_dist=True)

        self.target_val_acc(target_preds.argmax(dim=-1), target_labels)
        self.target_val_auroc(target_preds, target_labels)
        
        self.log('target_val_loss', target_loss, prog_bar=True, sync_dist=True)
        self.log('target_val_acc', self.target_val_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('target_val_auroc', self.target_val_auroc, on_step=True, on_epoch=True, sync_dist=True)

        loss = intensity_loss + target_loss
        return loss
    
    def test_step(self, batch, batch_idx): ## to be fixed
        labels = batch["labels"]
        
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )
        preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        
        self.test_acc(preds.argmax(dim=-1), labels)
        self.test_auroc(preds, labels)

        intensity_labels = batch['intensity_labels']
        target_labels = batch['target_labels']


        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )

        intensity_preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        target_preds = self.mlp2(model_outputs.multimodal_embeddings[:, 0])

        intensity_loss = F.cross_entropy(intensity_preds, intensity_labels) 
        target_loss = F.cross_entropy(target_preds, target_labels) 
        
        self.intensity_val_acc(intensity_preds.argmax(dim=-1), intensity_labels)
        self.intensity_val_auroc(intensity_preds, intensity_labels)

        self.target_val_acc(target_preds.argmax(dim=-1), target_labels)
        self.target_val_auroc(target_preds, target_labels)

        return None

    def on_test_epoch_end(self):
        print("intensity_test_acc:", self.intensity_test_acc.compute())
        print("intensity_test_auroc:", self.intensity_test_auroc.compute())

        print("target_test_acc:", self.target_test_acc.compute())
        print("target_test_auroc:", self.target_test_auroc.compute())

    def predict_step(self, batch, batch_idx): 
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )
        intensity_preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        target_preds = self.mlp2(model_outputs.multimodal_embeddings[:, 0])

        results = {"intensity_preds": intensity_preds}
        results = {"target_preds": target_preds}

        if "labels" in batch:
            results['labels'] = batch["labels"]

        return results
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return [self.optimizer]