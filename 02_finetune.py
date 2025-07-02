import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning import seed_everything

## import recformer library
from utils import read_json, AverageMeterSet, Ranker
from recformer import RecformerModel, RecformerForSeqRec, RecformerTokenizer, RecformerConfig
from collator import FinetuneDataCollatorWithPadding, EvalDataCollatorWithPadding
from dataloader import RecformerTrainDataset, RecformerEvalDataset

seed_everything(42)



def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, (1 - float(current_step) / float(max(1, num_training_steps)))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def create_optimizer_and_scheduler(model: nn.Module, num_train_optimization_steps, args):

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps=num_train_optimization_steps)

    return optimizer, scheduler
    
    


def load_data(args):

    train = read_json(args['train_file'], True)
    val = read_json(args['dev_file'], True)
    test = read_json(args['test_file'], True)
    item_meta_dict = json.load(open( args['meta_file'] ))
    
    item2id = read_json(args['item2id_file'])
    id2item = {v:k for k, v in item2id.items()}

    item_meta_dict_filted = dict()
    for k, v in item_meta_dict.items():
        if k in item2id:
            item_meta_dict_filted[k] = v

    return train, val, test, item_meta_dict_filted, item2id, id2item



def encode_all_items(model: RecformerModel, tokenizer: RecformerTokenizer, tokenized_items, args):

    model.eval()

    items = sorted(list(tokenized_items.items()), key=lambda x: x[0])
    items = [ele[1] for ele in items]

    item_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(items), args['batch_size']), ncols=100, desc='Encode all items', dynamic_ncols=True):

            item_batch = [[item] for item in items[i:i+args['batch_size']]]

            inputs = tokenizer.batch_encode(item_batch, encode_item=False)

            for k, v in inputs.items():
                inputs[k] = torch.LongTensor(v).to(args['device'])

            outputs = model(**inputs)

            item_embeddings.append(outputs.pooler_output.detach())

    item_embeddings = torch.cat(item_embeddings, dim=0)#.cpu()

    return item_embeddings




args = {
  "model_name_or_path": "../longformer-base-4096"
, "longformer_ckpt": '../longformer_ckpt/longformer-base-4096.bin'
, "pretrain_ckpt":"../pretrain_ckpt/recformer_seqrec_ckpt.bin"
, "ckpt":"best_model.bin"
    
, "train_file": "../finetune_data/Scientific/train.json"
, "dev_file": "../finetune_data/Scientific/val.json"
, "test_file": "../finetune_data/Scientific/test.json"
, "item2id_file": "../finetune_data/Scientific/smap.json"
, "meta_file" : "../finetune_data/Scientific/meta_data.json"

, "batch_size" : 4
, "finetune_negative_sample_size":-1
, "metric_ks":[10,50]
, "learning_rate":5e-5
, "weight_decay":0
, "warmup_steps":100
, "verbose":3
    
, "num_train_epochs": 2
, "gradient_accumulation_steps":8

, "preprocessing_num_workers" : 0
, "dataloader_num_workers": 0
, "device":0
, "fp16":True
, "fix_word_embedding":True
, "temp" :0.05

, "output_dir": "../result/recformer_finetune"
}

args['device'] = torch.device('cuda:{}'.format(args['device'])) if args['device']>=0 else torch.device('cpu')

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


## load raw data
train, val, test, item_meta_dict, item2id, id2item = load_data(args)

## load longformer tokenizer

config = RecformerConfig.from_pretrained(args['model_name_or_path'])
config.max_attr_num = 3
config.max_attr_length = 32
config.max_item_embeddings = 51
config.attention_window = [64] * 12
config.max_token_num = 1024
config.item_num = len(item2id)
config.finetune_negative_sample_size = args['finetune_negative_sample_size']

## load longformer tokenizer
tokenizer = RecformerTokenizer.from_pretrained(args['model_name_or_path'], config)  


## tokenize meta_data.json
path_tokenized_items = args['meta_file']+'.tokenized'

if os.path.exists(path_tokenized_items):
    print(f'[Preprocessor] Use cache: {path_tokenized_items}')
    tokenized_items = torch.load(path_tokenized_items)
else:
    tokenized_items = {}
    loop  = 0 
    
    for item_id, item_attr in item_meta_dict.items():
        input_ids, token_type_ids = tokenizer.encode_item(item_attr)
        tokenized_items[ item2id[item_id] ] = [input_ids, token_type_ids]
        if loop % 1000 == 0:
            print(time.ctime(), loop)
        loop+=1

    torch.save(tokenized_items, path_tokenized_items)

print("Item List:",len(tokenized_items)) # 5327




## load data
finetune_data_collator = FinetuneDataCollatorWithPadding(tokenizer, tokenized_items)
eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)

train_data = RecformerTrainDataset(train, collator=finetune_data_collator)
val_data = RecformerEvalDataset(train, val, test, mode='val', collator=eval_data_collator)
test_data = RecformerEvalDataset(train, val, test, mode='test', collator=eval_data_collator)


train_loader = DataLoader(train_data, 
                          batch_size=args['batch_size'], 
                          shuffle=True, 
                          collate_fn=train_data.collate_fn)
dev_loader = DataLoader(val_data, 
                        batch_size=args['batch_size'],
                        collate_fn=val_data.collate_fn)
test_loader = DataLoader(test_data, 
                        batch_size=args['batch_size'],
                        collate_fn=test_data.collate_fn)


## load checkpoint
model = RecformerForSeqRec(config)
pretrain_ckpt = torch.load(args['pretrain_ckpt'])
model.load_state_dict(pretrain_ckpt, strict=False)
model.to(args['device'])

if args['fix_word_embedding']:
    print('Fix word embeddings.')
    for param in model.longformer.embeddings.word_embeddings.parameters():
        param.requires_grad = False


## item embedding
path_item_embeddings =  args['meta_file']+'.embedding'

try:
    print(f'[Item Embeddings] Use cache: {path_tokenized_items}')
    item_embeddings = torch.load(path_item_embeddings)
except:
    print(f'Encoding items.')
    item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
    torch.save(item_embeddings, path_item_embeddings)

model.init_item_embedding(item_embeddings)
model.to(args['device'])

print(len(item_embeddings)) # 5327







def eval(model, dataloader, args):

    model.eval()

    ranker = Ranker(args['metric_ks'])
    average_meter_set = AverageMeterSet()

    for batch, labels in tqdm(dataloader, ncols=100, desc='Evaluate', dynamic_ncols=True):

        for k, v in batch.items():
            batch[k] = v.to(args['device'])
            
        labels = labels.to(args['device'])

        with torch.no_grad():
            scores = model(**batch)

        res = ranker(scores, labels)

        metrics = {}
        for i, k in enumerate(args['metric_ks']):
            metrics["NDCG@%d" % k] = res[2*i]
            metrics["Recall@%d" % k] = res[2*i+1]
        metrics["MRR"] = res[-3]
        metrics["AUC"] = res[-2]

        for k, v in metrics.items():
            average_meter_set.update(k, v)

    average_metrics = average_meter_set.averages()

    return average_metrics

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, args):

    model.train()

    for step, batch in enumerate(tqdm(dataloader, ncols=100, desc='Training', dynamic_ncols=True)):
        for k, v in batch.items():
            batch[k] = v.to(args['device'])

        if args['fp16']:
            with autocast():
                loss = model(**batch)
        else:
            loss = model(**batch)

        if args['gradient_accumulation_steps'] > 1:
            loss = loss / args['gradient_accumulation_steps']

        if args['fp16']:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args['gradient_accumulation_steps'] == 0:
            if args['fp16']:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                optimizer_was_run = scale_before <= scale_after
                optimizer.zero_grad()

                if optimizer_was_run:
                    scheduler.step()

            else:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                optimizer.zero_grad()





num_train_optimization_steps = int(len(train_loader) / args['gradient_accumulation_steps']) * args['num_train_epochs']
optimizer, scheduler = create_optimizer_and_scheduler(model, num_train_optimization_steps, args)

if args['fp16']:
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None

test_metrics = eval(model, test_loader, args)
print(f'Test set: {test_metrics}')

best_target = float('-inf')
patient = 5





for epoch in range(args['num_train_epochs']):

    item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
    model.init_item_embedding(item_embeddings)

    train_one_epoch(model, train_loader, optimizer, scheduler, scaler, args)
    
    if (epoch + 1) % args['verbose'] == 0:
        dev_metrics = eval(model, dev_loader, args)
        print(f'Epoch: {epoch}. Dev set: {dev_metrics}')

        if dev_metrics['NDCG@10'] > best_target:
            print('Save the best model.')
            best_target = dev_metrics['NDCG@10']
            patient = 5
            torch.save(model.state_dict(), args['ckpt'])
        
        else:
            patient -= 1
            if patient == 0:
                break

print('Load best model in stage 1.')
model.load_state_dict(torch.load(args['ckpt']))

patient = 3

for epoch in range(args['num_train_epochs']):

    train_one_epoch(model, train_loader, optimizer, scheduler, scaler, args)
    
    if (epoch + 1) % args['verbose'] == 0:
        dev_metrics = eval(model, dev_loader, args)
        print(f'Epoch: {epoch}. Dev set: {dev_metrics}')

        if dev_metrics['NDCG@10'] > best_target:
            print('Save the best model.')
            best_target = dev_metrics['NDCG@10']
            patient = 3
            torch.save(model.state_dict(), args['ckpt'])
        
        else:
            patient -= 1
            if patient == 0:
                break

print('Test with the best checkpoint.')  
model.load_state_dict(torch.load(args['ckpt']))
test_metrics = eval(model, test_loader, args)
print(f'Test set: {test_metrics}')
