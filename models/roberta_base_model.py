import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import RobertaForSequenceClassification

from model_utils.prompthate.classifier import SingleClassifier, SimpleClassifier
from model_utils.prompthate.rela_encoder import Rela_Module 

class RobertaBaseModel(nn.Module):
    def __init__(self,roberta,classifier,attention,proj_v):
        super(RobertaBaseModel, self).__init__()
        self.text_encoder=roberta
        self.classifier=classifier
        self.attention=attention
        self.proj_v=proj_v

    def forward(self,tokens,attention_mask,feat=None):
        output=self.text_encoder(tokens,
                                 attention_mask=attention_mask)[1][-1]
        if feat==None:
            joint_repre=output[:,0]
        else:
            #print ('Multimodal')
            text_repre=output[:,0]
            vis=self.proj_v(feat)
            att_vis=self.attention(vis,output)
            joint_repre=torch.cat((att_vis,text_repre),dim=1)
        logits=self.classifier(joint_repre)
        return logits
        
class ActualModel(pl.LightningModule):
    def __init__(self, model_class_or_path, opt):
        super().__init__()
        self.save_hyperparameters()
        final_dim=2
        times=2-int(opt["UNIMODAL"])
        text_encoder=RobertaForSequenceClassification.from_pretrained(
            model_class_or_path,
            num_labels=final_dim,
            output_attentions=False,
            output_hidden_states=True
        )
        attention=Rela_Module(opt["ROBERTA_DIM"],
                            opt["ROBERTA_DIM"],opt["NUM_HEAD"],opt["MID_DIM"],
                            opt["TRANS_LAYER"],
                            opt["FC_DROPOUT"])
        classifier=SimpleClassifier(opt["ROBERTA_DIM"]*times,
                                    opt["MID_DIM"],final_dim,opt["FC_DROPOUT"])
        proj_v=SingleClassifier(opt["FEAT_DIM"],opt["ROBERTA_DIM"],opt["FC_DROPOUT"])
        self.model = RobertaBaseModel(text_encoder,classifier,attention,proj_v)
       
    
    def training_step(self, batch, batch_idx):
        cap=batch['cap_tokens'].long().cuda()
        label=batch['label'].float().cuda().view(-1,1)
        mask=batch['mask'].cuda()
        target=batch['target'].cuda()
        feat=None
        logits=self.model(cap,mask,feat)

        loss = F.cross_entropy(logits, target)

        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return [self.optimizer]
    