import os
import random
import tqdm
import numpy as np
import pickle as pkl

from PIL import Image
from . import utils

from typing import List
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

class MultimodalDataBase():
    #mem, off, harm
    def __init__(self,opt,tokenizer,dataset,path, cap_path, prompt_arch, mode='train',few_shot_index=0):
        
        self.opt=opt
        self.tokenizer = tokenizer
        self.mode=mode
        if self.opt['FEW_SHOT']:
            self.few_shot_index=str(few_shot_index)
            self.num_shots=self.opt['NUM_SHOTS']
        
        self.num_ans=self.opt['NUM_LABELS']
        #maximum length for a single sentence
        self.length=self.opt['LENGTH']
        #maximum length of the concatenation of sentences
        self.total_length=self.opt['TOTAL_LENGTH']
        self.num_sample=self.opt['NUM_SAMPLE']
        self.add_ent=self.opt['ADD_ENT']
        self.add_dem=self.opt['ADD_DEM']
        self.fine_grind=self.opt['FINE_GRIND']
        self.prompt_arch = prompt_arch
        
        
        if self.opt['DEM_SAMP']:
            print ('Using demonstration sampling strategy...')
            self.img_rate=self.opt['IMG_RATE']
            self.text_rate=self.opt['TEXT_RATE']
            self.samp_rate=self.opt['SIM_RATE']
            
            self.clip_clean=self.opt['CLIP_CLEAN']
            clip_path=os.path.join(cap_path)
            
            self.clip_feature=utils.load_pkl(clip_path)
        
    def enc(self,text):
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def prepare_exp(self):
        ###add sampling
        support_indices = list(range(len(self.support_examples)))
        self.example_idx = []
        for sample_idx in tqdm(range(self.num_sample)):
            for query_idx in range(len(self.entries)):
                if self.opt['DEM_SAMP']:
                    #filter dissimilar demonstrations
                    candidates= [support_idx for support_idx in support_indices
                                 if support_idx != query_idx or self.mode != "train"]
                    sim_score=[]
                    count_each_label = {label: 0 for label in range(self.opt['NUM_LABELS'])}
                    context_indices=[]
                    clip_info_que=self.clip_feature[self.entries[query_idx]['img']]
                    
                    #similarity computation
                    for support_idx in candidates:
                        img=self.support_examples[support_idx]['img']
                        #this cost a lot of computation
                        #unnormalized: the same scale -- 512 dimension
                        if self.clip_clean:
                            img_sim=clip_info_que['clean_img'][img]
                        else:
                            img_sim=clip_info_que['img'][img]
                        text_sim=clip_info_que['text'][img]
                        total_sim=self.img_rate*img_sim+self.text_rate*text_sim
                        sim_score.append((support_idx,total_sim))
                    sim_score.sort(key=lambda x: x[1],reverse=True)
                    
                    #top opt['SIM_RATE'] entities for each label
                    num_valid=int(len(sim_score)//self.opt['NUM_LABELS']*self.samp_rate)
                    """
                    if self.opt['DEBUG']:
                        print ('Valid for each class:',num_valid)
                    """
                    
                    for support_idx, score in sim_score:
                        cur_label=self.support_examples[support_idx]['label']
                        if count_each_label[cur_label]<num_valid:
                            count_each_label[cur_label]+=1
                            context_indices.append(support_idx)
                else: 
                    #exclude the current example during training
                    context_indices = [support_idx for support_idx in support_indices
                                       if support_idx != query_idx or self.mode != "train"]
                #available indexes for supporting examples
                self.example_idx.append((query_idx, context_indices, sample_idx))

    def select_context(self, context_examples):
        """
        Select demonstrations from provided examples.
        """
        num_labels=self.opt['NUM_LABELS']
        max_demo_per_label = 1
        counts = {k: 0 for k in range(num_labels)}
        if num_labels == 1:
            # Regression
            counts = {'0': 0, '1': 0}
        selection = []
        """
        # Sampling strategy from LM-BFF
        if self.opt['DEBUG']:
            print ('Number of context examples available:',len(context_examples))
        """
        order = np.random.permutation(len(context_examples))
        for i in order:
            label = context_examples[i]['label']
            if num_labels == 1:
                # Regression
                #No implementation currently
                label = '0' if\
                float(label) <= median_mapping[self.args.task_name] else '1'
            if counts[label] < max_demo_per_label:
                selection.append(context_examples[i])
                counts[label] += 1
            if sum(counts.values()) == len(counts) * max_demo_per_label:
                break
        
        assert len(selection) > 0
        return selection
    
    def process_prompt(self, examples, 
                       first_sent_limit, other_sent_limit):
        prompt_arch = self.prompt_arch
        #currently, first and other limit are the same
        input_ids = []
        attention_mask = []
        mask_pos = None # Position of the mask token
        concat_sent=""
        for segment_id, ent in enumerate(examples):
            #tokens for each example
            new_tokens=[]
            if segment_id==0:
                #implementation for the querying example
                new_tokens.append(self.special_token_mapping['<s>'])
                length=first_sent_limit
                temp=prompt_arch+'<mask>'+' . </s>'
            else:
                length=other_sent_limit
                if self.fine_grind:
                    if ent['label']==0:
                        label_word=self.label_mapping_word[0]
                    else:
                        attack_types=[i for i, x in enumerate(ent['attack']) if x==1]
                        #only for meme
                        if len(attack_types)==0:
                            attack_idx=random.randint(1,5)
                        #randomly pick one
                        #already padding nobody to the head of the list
                        else:
                            order=np.random.permutation(len(attack_types))
                            attack_idx=attack_types[order[0]]
                        label_word=self.label_mapping_word[attack_idx]
                else:
                    label_word=self.label_mapping_word[ent['label']]
                temp=prompt_arch+label_word+' . </s>'
            new_tokens+=self.enc(' '+ent['cap'])
            #truncate the sentence if too long
            new_tokens=new_tokens[:length]
            new_tokens+=self.enc(temp)
            whole_sent=' '+ent['cap']+temp
            concat_sent+=whole_sent
        
            #update the prompts
            input_ids+=new_tokens
            attention_mask += [1 for i in range(len(new_tokens))]
        """
        if self.opt['DEBUG'] and self.opt['DEM_SAMP']==False:
            print (concat_sent)
        """
        while len(input_ids) < self.total_length:
            input_ids.append(self.special_token_mapping['<pad>'])
            attention_mask.append(0)
        if len(input_ids) > self.total_length:
            input_ids = input_ids[:self.total_length]
            attention_mask = attention_mask[:self.total_length]
        mask_pos = [input_ids.index(self.special_token_mapping['<mask>'])]
        
        # Make sure that the masked position is inside the max_length
        assert mask_pos[0] < self.total_length
        result = {'input_ids': input_ids,
                  'sent':'<s>'+concat_sent,
                  'attention_mask': attention_mask,
                  'mask_pos': mask_pos}
        return result
       
    def support_getitem(self,index):
        #query item
        entry=self.entries[index]
        #bootstrap_idx --> sample_idx
        query_idx, context_indices, bootstrap_idx = self.example_idx[index]
        #one example from each class
        supports = self.select_context(
            [self.support_examples[i] for i in context_indices])
        exps=[]
        exps.append(entry)
        exps.extend(supports)
        prompt_features = self.process_prompt(
            exps,
            self.length,
            self.length
        )
            
        vid=entry['img']
        #label=torch.tensor(self.label_mapping_id[entry['label']])
        label=torch.tensor(entry['label'])
        target=torch.from_numpy(np.zeros((self.num_ans),dtype=np.float32))
        target[label]=1.0
        
        cap_tokens=torch.Tensor(prompt_features['input_ids'])
        mask_pos=torch.LongTensor(prompt_features['mask_pos'])
        mask=torch.Tensor(prompt_features['attention_mask'])
        batch={
            'sent':prompt_features['sent'],
            'mask':mask,
            'img':vid,
            'target':target,
            'cap_tokens':cap_tokens,
            'mask_pos':mask_pos,
            'label':label
        }
        return batch, entry
        
    def __len__(self):
        return len(self.entries)