import json
import csv
import os
import torch.nn as nn
import torch
import pickle as pkl
import numpy as np
import torch
import lightning.pytorch as pl
from tqdm import tqdm
import pandas as pd

def _load_jsonl(filepath):
    data = []
    
    with open(filepath, 'r')as f:
        for line in f.readlines():
            obj = json.loads(line)
            data.append(obj)
    
    return data

def _load_csv(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            data.append(row)
    return data


def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def read_csv(path):
    data=pd.read_csv(path)
    return data

def read_json(path):
    assert_exits(path)
    data=json.load(open(path,'rb'))
    '''in anet-qa returns a list'''
    return data

class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self,prob,logits):
        #length = (prob.sum(-1) > 0.001).sum()
        #print (prob.shape,logits.shape)
        bz,obj=prob.shape
        length=bz
        #prob=torch.softmax(prob,-1)
        #pred_prob = torch.softmax(logits, -1)
        pred_prob=logits
        #print (prob.sum(),pred_prob.sum())
        print ('external:',pred_prob[0],
               torch.sort(pred_prob[0],
                          dim=0,
               descending=True)[1][0])
        print ('visual:',prob[0],
               torch.sort(prob[0],
                          dim=0,
               descending=True)[1][0])
        loss = -prob * torch.log(pred_prob)
        loss = torch.sum(loss, -1)
        loss = torch.sum(loss) / length
        return loss
    
def assert_exits(path):
    assert os.path.exists(path), 'Does not exist : {}'.format(path)
    
def equal_info(a,b):
    assert len(a)==len(b),'File info not equal!'
    
def same_question(a,b):
    assert a==b,'Not the same question!'
    
class Logger(object):
    def __init__(self,output_dir):
        dirname=os.path.dirname(output_dir)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        self.log_file=open(output_dir,'w')
        self.infos={}
        
    def append(self,key,val):
        vals=self.infos.setdefault(key,[])
        vals.append(val)

    def log(self,extra_msg=''):
        msgs=[extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' %(key,np.mean(vals)))
        msg='\n'.joint(msgs)
        self.log_file.write(msg+'\n')
        self.log_file.flush()
        self.infos={}
        return msg
        
    def write(self,msg):
        self.log_file.write(msg+'\n')
        self.log_file.flush()
        print(msg)    