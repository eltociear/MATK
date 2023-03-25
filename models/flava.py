import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics


from transformers import FlavaModel

class FlavaClassificationModel(pl.LightningModule):
    def __init__(self, model_class_or_path, shaming_classes=2, misogynous_classes=2, stereotype_classes=2, objectification_classes=2, violence_classes=2):
        super().__init__()
        self.save_hyperparameters()
        self.model = FlavaModel.from_pretrained(model_class_or_path)
        self.mlp = nn.Sequential(
            nn.Linear(self.model.config.multimodal_config.hidden_size, shaming_classes)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(self.model.config.multimodal_config.hidden_size, misogynous_classes)
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(self.model.config.multimodal_config.hidden_size, stereotype_classes)
        )
        self.mlp4 = nn.Sequential(
            nn.Linear(self.model.config.multimodal_config.hidden_size, objectification_classes)
        )
        self.mlp5 = nn.Sequential(
            nn.Linear(self.model.config.multimodal_config.hidden_size, violence_classes)
        )
        
        #shaming
        self.shaming_train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=shaming_classes)
        self.shaming_train_auroc = torchmetrics.AUROC(task="multiclass", num_classes=shaming_classes)

        self.shaming_val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=shaming_classes)
        self.shaming_val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=shaming_classes)

        #misogynous
        self.misogynous_train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=misogynous_classes)
        self.misogynous_train_auroc = torchmetrics.AUROC(task="multiclass", num_classes=misogynous_classes)

        self.misogynous_val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=misogynous_classes)
        self.misogynous_val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=misogynous_classes)

        #stereotype
        self.stereotype_train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=stereotype_classes)
        self.stereotype_train_auroc = torchmetrics.AUROC(task="multiclass", num_classes=stereotype_classes)

        self.stereotype_val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=stereotype_classes)
        self.stereotype_val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=stereotype_classes)

        #objectification
        self.objectification_train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=objectification_classes)
        self.objectification_train_auroc = torchmetrics.AUROC(task="multiclass", num_classes=objectification_classes)

        self.objectification_val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=objectification_classes)
        self.objectification_val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=objectification_classes)

        #violence
        self.violence_train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=violence_classes)
        self.violence_train_auroc = torchmetrics.AUROC(task="multiclass", num_classes=violence_classes)

        self.violence_val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=violence_classes)
        self.violence_val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=violence_classes)


        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)

        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_auroc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)
    
    def training_step(self, batch, batch_idx):
        shaming_labels = batch['shaming']

        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )
        shaming_preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        shaming_loss = F.cross_entropy(shaming_preds, shaming_labels)
        
        self.shaming_train_acc(shaming_preds.argmax(dim=-1), shaming_labels)
        self.shaming_train_auroc(shaming_preds, shaming_labels)
        self.log('shaming_train_loss', shaming_loss, prog_bar=True)
        self.log('shaming_train_acc', self.shaming_train_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('shaming_train_auroc', self.shaming_train_auroc, on_step=True, on_epoch=True, sync_dist=True)

        misogynous_labels = batch['misogynous']

        misogynous_preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        misogynous_loss = F.cross_entropy(misogynous_preds, misogynous_labels)
        
        self.misogynous_train_acc(misogynous_preds.argmax(dim=-1), misogynous_labels)
        self.misogynous_train_auroc(misogynous_preds, misogynous_labels)
        self.log('misogynous_train_loss', misogynous_loss, prog_bar=True)
        self.log('misogynous_train_acc', self.misogynous_train_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('misogynous_train_auroc', self.misogynous_train_auroc, on_step=True, on_epoch=True, sync_dist=True)

        stereotype_labels = batch['stereotype']

        stereotype_preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        stereotype_loss = F.cross_entropy(stereotype_preds, stereotype_labels)
        
        self.stereotype_train_acc(stereotype_preds.argmax(dim=-1), stereotype_labels)
        self.stereotype_train_auroc(stereotype_preds, stereotype_labels)
        self.log('stereotype_train_loss', stereotype_loss, prog_bar=True)
        self.log('stereotype_train_acc', self.stereotype_train_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('stereotype_train_auroc', self.stereotype_train_auroc, on_step=True, on_epoch=True, sync_dist=True)

        objectification_labels = batch['objectification']

        objectification_preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        objectification_loss = F.cross_entropy(objectification_preds, objectification_labels)
        
        self.objectification_train_acc(objectification_preds.argmax(dim=-1), objectification_labels)
        self.objectification_train_auroc(objectification_preds, objectification_labels)
        self.log('objectification_train_loss', objectification_loss, prog_bar=True)
        self.log('objectification_train_acc', self.objectification_train_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('objectification_train_auroc', self.objectification_train_auroc, on_step=True, on_epoch=True, sync_dist=True)

        violence_labels = batch['violence']

        violence_preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        violence_loss = F.cross_entropy(violence_preds, violence_labels)
        
        self.violence_train_acc(violence_preds.argmax(dim=-1), violence_labels)
        self.violence_train_auroc(violence_preds, violence_labels)
        self.log('violence_train_loss', violence_loss, prog_bar=True)
        self.log('violence_train_acc', self.violence_train_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('violence_train_auroc', self.violence_train_auroc, on_step=True, on_epoch=True, sync_dist=True)

        loss = shaming_loss + misogynous_loss + stereotype_loss + objectification_loss + violence_loss
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        shaming_labels = batch['shaming']

        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )
        shaming_preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        shaming_loss = F.cross_entropy(shaming_preds, shaming_labels)
        
        self.shaming_val_acc(shaming_preds.argmax(dim=-1), shaming_labels)
        self.shaming_val_auroc(shaming_preds, shaming_labels)
        self.log('shaming_val_loss', shaming_loss, prog_bar=True, sync_dist=True)
        self.log('shaming_val_acc', self.shaming_val_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('shaming_val_auroc', self.shaming_val_auroc, on_step=True, on_epoch=True, sync_dist=True)

        misogynous_labels = batch['misogynous']

        misogynous_preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        misogynous_loss = F.cross_entropy(misogynous_preds, misogynous_labels)
        
        self.misogynous_val_acc(misogynous_preds.argmax(dim=-1), misogynous_labels)
        self.misogynous_val_auroc(misogynous_preds, misogynous_labels)
        self.log('misogynous_val_loss', misogynous_loss, prog_bar=True, sync_dist=True)
        self.log('misogynous_val_acc', self.misogynous_val_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('misogynous_val_auroc', self.misogynous_val_auroc, on_step=True, on_epoch=True, sync_dist=True)

        stereotype_labels = batch['stereotype']

        stereotype_preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        stereotype_loss = F.cross_entropy(stereotype_preds, stereotype_labels)
        
        self.stereotype_val_acc(stereotype_preds.argmax(dim=-1), stereotype_labels)
        self.stereotype_val_auroc(stereotype_preds, stereotype_labels)
        self.log('stereotype_val_loss', stereotype_loss, prog_bar=True, sync_dist=True)
        self.log('stereotype_val_acc', self.stereotype_val_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('stereotype_val_auroc', self.stereotype_val_auroc, on_step=True, on_epoch=True, sync_dist=True)

        objectification_labels = batch['objectification']

        objectification_preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        objectification_loss = F.cross_entropy(objectification_preds, objectification_labels)
        
        self.objectification_val_acc(objectification_preds.argmax(dim=-1), objectification_labels)
        self.objectification_val_auroc(objectification_preds, objectification_labels)
        self.log('objectification_val_loss', objectification_loss, prog_bar=True, sync_dist=True)
        self.log('objectification_val_acc', self.objectification_val_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('objectification_val_auroc', self.objectification_val_auroc, on_step=True, on_epoch=True, sync_dist=True)

        violence_labels = batch['violence']

        violence_preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        violence_loss = F.cross_entropy(violence_preds, violence_labels)
        
        self.violence_val_acc(violence_preds.argmax(dim=-1), violence_labels)
        self.violence_val_auroc(violence_preds, violence_labels)
        self.log('violence_val_loss', violence_loss, prog_bar=True, sync_dist=True)
        self.log('violence_val_acc', self.violence_val_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('violence_val_auroc', self.violence_val_auroc, on_step=True, on_epoch=True, sync_dist=True)

        loss = shaming_loss + misogynous_loss + stereotype_loss + objectification_loss + violence_loss
        return loss
    
    def test_step(self, batch, batch_idx):
        labels = batch["labels"]
        
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )
        preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        
        self.test_acc(preds.argmax(dim=-1), labels)
        self.test_auroc(preds, labels)

        return None

    def on_test_epoch_end(self):
        print("test_acc:", self.test_acc.compute())
        print("test_auroc:", self.test_auroc.compute())

    def predict_step(self, batch, batch_idx):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )
        preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        results = {"preds": preds}

        if "labels" in batch:
            results['labels'] = batch["labels"]

        return results
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return [self.optimizer]