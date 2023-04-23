import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import RobertaForSequenceClassification

from model_utils.prompthate.classifier import SingleClassifier, SimpleClassifier
from model_utils.prompthate.rela_encoder import Rela_Module 

class RobertaBaseModel(pl.LightningModule):
    def __init__(self,model_class_or_path, opt, num_classes=2):
        # super(RobertaBaseModel, self).__init__()
        super().__init__()
        self.save_hyperparameters()
        final_dim=2
        times=2-int(opt.UNIMODAL)

        self.text_encoder=RobertaForSequenceClassification.from_pretrained(
            model_class_or_path,
            num_labels=final_dim,
            output_attentions=False,
            output_hidden_states=True
        )
        self.classifier=SimpleClassifier(opt.ROBERTA_DIM*times,
                                    opt.MID_DIM,final_dim,opt.FC_DROPOUT)
        self.attention=Rela_Module(opt.ROBERTA_DIM,
                            opt.ROBERTA_DIM,opt.NUM_HEAD,opt.MID_DIM,
                            opt.TRANS_LAYER,
                            opt.FC_DROPOUT)
        self.proj_v=SingleClassifier(opt.FEAT_DIM,opt.ROBERTA_DIM,opt.FC_DROPOUT)
        

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
        
    
    