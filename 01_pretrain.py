#!/usr/bin/env python
# coding: utf-8

import os,sys
import time
import logging
import json
from typing import Optional, Union, List, Dict, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

## import recformer library
from recformer import RecformerForPretraining, RecformerTokenizer, RecformerConfig, LitWrapper
from collator import PretrainDataCollatorWithPadding
from dataloader import PretrainDataset

seed_everything(42)






## arguments

args = {
  "model_name_or_path": "../longformer-base-4096"
, "longformer_ckpt": '../longformer_ckpt/longformer-base-4096.bin'

, "train_file": "../pretrain_data/train.json"  
, "dev_file": "../pretrain_data/dev.json"  
, "item_attr_file" : "../pretrain_data/meta_data.json"

, "batch_size" : 4
, "learning_rate":5e-5
, "num_train_epochs": 32
, "mlm_probability": 0.15
, "gradient_accumulation_steps":8

, "valid_step":2000
, "log_step":2000
, "device":1
, "dataloader_num_workers": 2
, "fp16":True
, "fix_word_embedding":True
, "temp" :0.05

, "output_dir": "../result/recformer_pretraining"
}




## load longformer tokenizer
config = RecformerConfig.from_pretrained(args['model_name_or_path'])
config.max_attr_num = 3
config.max_attr_length = 32
config.max_item_embeddings = 51  # 50 item and 1 for cls
config.attention_window = [64] * 12
config.max_token_num = 1024

tokenizer = RecformerTokenizer.from_pretrained(args['model_name_or_path'], config)  

# tokenizer._pad_token  = tokenizer.pad_token_id



## tokenize meta_data.json
path_tokenized_items = args['item_attr_file']+'.tokenized'

if os.path.exists(path_tokenized_items):
    print(f'[Preprocessor] Use cache: {path_tokenized_items}')
    tokenized_items = torch.load(path_tokenized_items)

else:
    item_attrs = json.load(open( args['item_attr_file'] ))
    
    tokenized_items = {}
    for item_id, item_attr in item_attrs.items():
        input_ids, token_type_ids = tokenizer.encode_item(item_attr)
        tokenized_items[ item_id ] = [input_ids, token_type_ids]

    torch.save(tokenized_items, path_tokenized_items)




## load data

data_collator = PretrainDataCollatorWithPadding(tokenizer, tokenized_items, mlm_probability=args['mlm_probability'])
train_data = PretrainDataset(json.load(open(args['train_file'])), data_collator)
dev_data = PretrainDataset(json.load(open(args['dev_file'])), data_collator)

train_loader = DataLoader(train_data, 
                          batch_size=args['batch_size'], 
                          shuffle=True, 
                          collate_fn=train_data.collate_fn,
                          num_workers=args['dataloader_num_workers'])

dev_loader = DataLoader(dev_data, 
                        batch_size=args['batch_size'], 
                        collate_fn=dev_data.collate_fn,
                        num_workers=args['dataloader_num_workers'])


## load checkpoint
pytorch_model = RecformerForPretraining(config)
pytorch_model.load_state_dict(torch.load(args['longformer_ckpt']))

if args['fix_word_embedding']:
    print('Fix word embeddings.')
    for param in pytorch_model.longformer.embeddings.word_embeddings.parameters():
        param.requires_grad = False



model = LitWrapper(pytorch_model, learning_rate=args['learning_rate'])

checkpoint_callback = ModelCheckpoint(save_top_k=5, monitor="accuracy", mode="max", filename="{epoch}-{accuracy:.4f}")

trainer = Trainer(accelerator="gpu",
                 max_epochs=args['num_train_epochs'],
                 devices=args['device'],
                 accumulate_grad_batches=args['gradient_accumulation_steps'],
                 val_check_interval=args['valid_step'],
                 default_root_dir=args['output_dir'],
                 gradient_clip_val=1.0,
                 log_every_n_steps=args['log_step'],
                 precision=16 if args['fp16'] else 32,
                 strategy='ddp',
                 callbacks=[checkpoint_callback]
                 )

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=dev_loader)


