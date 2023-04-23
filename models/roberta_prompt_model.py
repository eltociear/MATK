import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import RobertaForMaskedLM

class RobertaPromptModel(pl.LightningModule):
    def __init__(self, model_class_or_path, label_list):
        super(RobertaPromptModel, self).__init__()
        self.label_word_list=label_list
        self.roberta = RobertaForMaskedLM.from_pretrained(model_class_or_path)

    def forward(self,tokens,attention_mask,mask_pos,feat=None):
        batch_size = tokens.size(0)
        #the position of word for prediction
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()
            
        out = self.roberta(tokens, 
                           attention_mask)
        prediction_mask_scores = out.logits[torch.arange(batch_size),
                                          mask_pos]
        
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:,
                                                 self.label_word_list[label_id]
                                                ].unsqueeze(-1))
            #print(prediction_mask_scores[:, self.label_word_list[label_id]].shape)
        logits = torch.cat(logits, -1)
        #print(logits.shape)
        return logits
        

class ActualModel(pl.LightningModule):
    def __init__(self, model_class_or_path, label_list):
        super().__init__()
        self.save_hyperparameters()
        self.model = RobertaPromptModel(model_class_or_path, label_list)
       
    
    def training_step(self, batch, batch_idx):
        cap=batch['cap_tokens'].long().cuda()
        label=batch['label'].float().cuda().view(-1,1)
        mask=batch['mask'].cuda()
        target=batch['target'].cuda()
        feat=None
        mask_pos=batch['mask_pos'].cuda()
        logits=self.model(cap,mask,mask_pos,feat)

        loss = F.cross_entropy(logits, target)

        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return [self.optimizer]