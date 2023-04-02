import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics


from transformers import LxmertModel
# from ..datamodules.VL_T5_main.VL_T5.src.modeling_t5 import VLT5
from .utilt5 import VLT5
from transformers import T5Config, BartConfig
 
from .param import parse_args

def create_config():

    args = parse_args( 
        backbone='t5-base', # Backbone architecture
        load='Epoch30.pth', # Pretrained checkpoint 
        parse=False, # False for interactive env (ex. jupyter)
    )
    args.gpu = 0 # assign GPU 
    # create config file 
    
    print("args")
    print(args)
    config_class = T5Config 
    config = config_class.from_pretrained('t5-base')
    config.feat_dim = args.feat_dim
    config.pos_dim = args.pos_dim
    config.n_images = 2

    config.use_vis_order_embedding = args.use_vis_order_embedding

    config.dropout_rate = args.dropout
    config.dropout = args.dropout
    config.attention_dropout = args.dropout
    config.activation_dropout = args.dropout

    config.use_vis_layer_norm = args.use_vis_layer_norm
    config.individual_vis_layer_norm = args.individual_vis_layer_norm
    config.losses = args.losses

    config.share_vis_lang_layer_norm = args.share_vis_lang_layer_norm
    config.classifier = args.classifier
    return config
        
class VLT5ClassificationModel(pl.LightningModule):
    def __init__(self, model_class_or_path, num_classes=2):
        super().__init__()
        self.save_hyperparameters()
        config = create_config()
        self.model = VLT5.from_pretrained(model_class_or_path, config=config)
        self.model.resize_token_embeddings(32200)
        
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_auroc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)

        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)

        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_auroc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)

        
    def training_step(self, batch, batch_idx):
        labels = batch['labels']

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        visual_feats = batch['visual_feats']
        visual_pos = batch['visual_pos']
        token_type_ids = batch['token_type_ids']
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vis_inputs=(visual_feats, visual_pos),
            token_type_ids = token_type_ids
        )

        pooled_output = outputs[0][:, 0]  # Extract the [CLS] token embedding
        preds = self.classifier(pooled_output)
        loss = F.cross_entropy(preds, labels)

        self.train_acc(preds.argmax(dim=-1), labels)
        self.train_auroc(preds, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_auroc', self.train_auroc, on_step=True, on_epoch=True, sync_dist=True)


        return loss

    
    def validation_step(self, batch, batch_idx):
        labels = batch['labels']

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        visual_feats = batch['visual_feats']
        visual_pos = batch['visual_pos']
        token_type_ids = batch['token_type_ids']
        
        output = self.model(
            input_ids=input_ids,
            vis_inputs=(visual_feats, visual_pos),
            labels=labels,
            reduce_loss=True
        )

        pooled_output = outputs[0][:, 0]  # Extract the [CLS] token embedding
        preds = self.classifier(pooled_output)
        loss = F.cross_entropy(preds, labels)

        self.train_acc(preds.argmax(dim=-1), labels)
        self.train_auroc(preds, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_auroc', self.train_auroc, on_step=True, on_epoch=True, sync_dist=True)


        return loss

    
    def test_step(self, batch, batch_idx):
        labels = batch['labels']

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        visual_feats = batch['visual_feats']
        visual_pos = batch['visual_pos']
        token_type_ids = batch['token_type_ids']
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vis_inputs=(visual_feats, visual_pos),
            token_type_ids = token_type_ids
        )

        pooled_output = outputs[0][:, 0]  # Extract the [CLS] token embedding
        preds = self.classifier(pooled_output)
        loss = F.cross_entropy(preds, labels)

        self.train_acc(preds.argmax(dim=-1), labels)
        self.train_auroc(preds, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_auroc', self.train_auroc, on_step=True, on_epoch=True, sync_dist=True)


        return loss


    def on_test_epoch_end(self):
        print("test_acc:", self.test_acc.compute())
        print("test_auroc:", self.test_auroc.compute())

    def predict_step(self, batch, batch_idx):
        labels = batch['labels']

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        visual_feats = batch['visual_feats']
        visual_pos = batch['visual_pos']
        token_type_ids = batch['token_type_ids']
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vis_inputs=(visual_feats, visual_pos),
            token_type_ids = token_type_ids
        )

        pooled_output = outputs[0][:, 0]  # Extract the [CLS] token embedding
        preds = self.classifier(pooled_output)
        loss = F.cross_entropy(preds, labels)

        self.train_acc(preds.argmax(dim=-1), labels)
        self.train_auroc(preds, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_auroc', self.train_auroc, on_step=True, on_epoch=True, sync_dist=True)


        return loss

    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return [self.optimizer]