######## IMPORTS
import os
import sys

from time import time
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn


from sklearn.mixture import GaussianMixture
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from logger import FileLogger, TensorboardLogger
from models import model_builder

from config import load_config, update_params
from torch.autograd import Variable

from utils.functions import set_random_seed
from utils.datasets import load_pair_datasets, load_datasets, load_biencodier_pairwise_datasets
from utils.evaluation import evaluate_biencoder, evaluate_biencoder_qr

os.environ["TOKENIZERS_PARALLELISM"] = "true"

conf = load_config()

#####################################################
#                  Update params                    #
#####################################################
argv = sys.argv[1:]
if len(argv) > 0:
    cmd_arg = OrderedDict()
    argvs = ' '.join(sys.argv[1:]).split(' ')
    for i in range(0, len(argvs), 2):
        arg_name, arg_value = argvs[i], argvs[i + 1]
        arg_name = arg_name.strip('-')
        cmd_arg[arg_name] = arg_value
    conf = update_params(conf, cmd_arg)

gpu = str(conf.base.gpus)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = "cuda:%s" % conf.base.gpus if torch.cuda.is_available() else "cpu"

set_random_seed(conf.base.seed)

#####################################################
#                   Load BERT                       #
#####################################################
print(f'Load BERT model... {conf.model.bert_model}')


print(f'> You are now using HuggingFace library.')
from transformers import AutoTokenizer, AutoModel
bertmodel = AutoModel.from_pretrained(conf.model.bert_model)
tokenizer = AutoTokenizer.from_pretrained(conf.model.bert_model)

#####################################################
#                   Load dataset                    #
#####################################################


train_file = os.path.join(conf.dataset.datadir, 'sample/train.tsv')
valid_file = os.path.join(conf.dataset.datadir, 'sample/valid.tsv')
test_file = os.path.join(conf.dataset.datadir, 'sample/test.tsv')


print('Load dataset...')

if conf.model.pairwise:
    train_dataset, _, _ = load_biencodier_pairwise_datasets(
                                                trainfile=train_file,
                                                bert_tokenizer=tokenizer, 
                                                ori_idx=0,
                                                reduced_idx=1,
                                                **conf.dataset)
    _, valid_dataset, _ = load_pair_datasets(
                                                validfile=valid_file,
                                                bert_tokenizer=tokenizer, 
                                                ori_idx=0,
                                                reduced_idx=1,
                                                **conf.dataset)  

else:
    train_dataset, valid_dataset, _ = load_pair_datasets(
                                                    trainfile=train_file,
                                                    validfile=valid_file,
                                                    bert_tokenizer=tokenizer, 
                                                    ori_idx=0,
                                                    reduced_idx=1,
                                                    **conf.dataset)                       

_, _, test_dataset = load_datasets(
    validfile=None,
    testfile=test_file,
    bert_tokenizer=tokenizer, 
    ori_idx=0,
    reduced_idx=1,  
    **conf.dataset)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.exp.batch_size, num_workers=10, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=conf.exp.batch_size, num_workers=10)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.exp.batch_size, num_workers=10)


#####################################################
#                   Build BERT                      #
#####################################################
print('Build sentence pair classification model...')
model_name = 'bert_tree' if conf.model.tree_transformer else 'bert'
model = model_builder(model_name, bert=bertmodel, device=device, **conf.model).to(device)



#####################################################
#                 Set up training                   #
#####################################################
# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]


## Setting parameters
optimizer = AdamW(optimizer_grouped_parameters, lr=conf.exp.learning_rate)


## pair-wise loss

if conf.model.pairwise:
    criterion = nn.CrossEntropyLoss(reduction='none')
else:
    criterion = nn.BCEWithLogitsLoss(reduction='none')


t_total = len(train_dataloader) * conf.exp.num_epochs
warmup_step = int(t_total * conf.exp.warmup_ratio)

if conf.exp.truncated_loss:
    if conf.exp.forget_rate is None:
        forget_rate=0.1
    else:
        forget_rate = conf.exp.forget_rate

    rate_schedule = np.ones(conf.exp.num_epochs)*forget_rate
    rate_schedule[:conf.exp.num_gradual] = np.linspace(0, forget_rate**conf.exp.exponent, conf.exp.num_gradual)

if warmup_step > 0:
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
else:
    scheduler = None

#### File logger: log hparams, evaluation results into a csv file in 'log_dir'.
file_logger = FileLogger(conf.base.save_dir, conf.base.exp_name)
log_dir = file_logger.log_dir
print(f'log & save model in {log_dir}...')

#### Tensorboard logger
if conf.base.tensorboard:
    tensorboard = TensorboardLogger(
        log_dir=log_dir,
        experiment_name=conf.base.exp_name,
        hparams=dict(conf),
        log_graph=False
    )
else:
    tensorboard = None

def train_epoch(model, train_dataloader, criterion, optimizer, scheduler, max_grad_norm=0.0, verbose=0, device='cpu',forget_rate=0.1):
    model.train()
    epoch_loss = 0.0
    elapsed = {
        'data': 0.0,
        'forward': 0.0,
        'backward': 0.0,
        'step': 0.0
    }

    for batch_id, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
        optimizer.zero_grad()
        
        token_ids = batch['token_ids'].long().to(device)
        valid_length = batch['valid_length']
        segment_ids = batch['segment_ids'].long().to(device)
        types = batch['type']
        



        if conf.model.pairwise:
            pos_token_ids = batch['pos_tok'].long().to(device)
            pos_valid_length = batch['pos_valid_length']
            pos_segment_ids = batch['pos_segment_ids'].long().to(device)
            ori_token_ids = batch['original_tok'].long().to(device)
            ori_valid_length = batch['original_valid_length']
            ori_segment_ids = batch['original_segment_ids'].long().to(device)
            
            ori_model_out = model(ori_token_ids, ori_valid_length, ori_segment_ids)
            pos_model_out = model(pos_token_ids, pos_valid_length, pos_segment_ids)
            ori_cls_out = ori_model_out['cls_out']
            pos_cls_out = pos_model_out['cls_out']
            
            
            for i in range(conf.dataset.num_negatives):
                neg_token_ids = batch['neg_tok'][i].long().to(device)
                neg_valid_length = batch['neg_valid_length'][i]
                neg_segment_ids = batch['neg_segment_ids'][i].long().to(device)
                neg_model_out = model(neg_token_ids, neg_valid_length, neg_segment_ids)
                
                if i==0:
                    neg_output = torch.diag(ori_cls_out @ neg_model_out['cls_out'].T)
                    
                    neg_output = neg_output.view(len(neg_output),1)
                else:
                    neg_output = torch.cat([neg_output, torch.diag(ori_cls_out @ neg_model_out['cls_out'].T).view(len(neg_output),1)],dim=1)
                

            pos_output = torch.diag(ori_cls_out @ pos_cls_out.T)


            
            pair_out = torch.cat([pos_output.view(len(pos_output),1),neg_output],dim=1)

            tmp_label = torch.zeros(len(pos_output), dtype=torch.long, device = device)
            _loss = criterion(pair_out, tmp_label)            

            
        else:
            label = batch['label'].to(device)
            
            original_tok = batch['original_tok'].long().to(device)
            original_valid_length = batch['original_valid_length']
            original_segment_ids = batch['original_segment_ids'].long().to(device)
            reduced_tok = batch['reduced_tok'].long().to(device)
            reduced_valid_length = batch['reduced_valid_length']
            reduced_segment_ids = batch['reduced_segment_ids'].long().to(device)

            model_out = model(original_tok, original_valid_length, original_segment_ids)
            cls_out = model_out['cls_out']
            model_out2 = model(reduced_tok, reduced_valid_length, reduced_segment_ids)
            cls_out2 = model_out2['cls_out']

            output = torch.sum(cls_out * cls_out2)

            _loss = criterion(output,label)

        

        if conf.exp.truncated_loss:

            if conf.model.pairwise:
                idx_loss_sorted = np.argsort(_loss.data.cpu())
                _loss_sorted = _loss[idx_loss_sorted]
                remember_rate = 1 - forget_rate
                num_remember = int(remember_rate * len(_loss_sorted))
                idx_update = idx_loss_sorted[:num_remember]
                truncated_pair_out = pair_out[idx_update]
                tmp_label = torch.zeros(len(truncated_pair_out), dtype=torch.long, device = device)
                loss_update = criterion(truncated_pair_out, tmp_label)
                loss = loss_update.mean()

            else:
                idx_loss_sorted = np.argsort(_loss.data.cpu())
                _loss_sorted = _loss[idx_loss_sorted]
                remember_rate = 1 - forget_rate
                num_remember = int(remember_rate * len(_loss_sorted))
                idx_update = idx_loss_sorted[:num_remember]
                idx_update = idx_update.tolist()
                loss_update = criterion(output[idx_update],label[idx_update])
                
                loss = loss_update.mean()

                
                
        else:
            loss = _loss.mean()
        
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        epoch_loss += loss.item()

        if verbose > 0 and (batch_id + 1) % verbose == 0:
            print('\tbatch %d loss:' % (batch_id+1), loss.item())
    return epoch_loss


################################ MAIN
if __name__ == "__main__":
    try:
        file_logger.log_hparams(dict(conf))
        file_logger.save_hparams()

        print('Training begins...')
        best_em = -1
        best_epoch = -1
        best_ckpt = None
        for epoch in range(1, conf.exp.num_epochs + 1):
            epoch_train_dataloader = train_dataloader

            # Training phase
            train_start = time()
            # train for one epoch
            if conf.exp.truncated_loss:
                epoch_loss = train_epoch(model, epoch_train_dataloader, criterion, optimizer, scheduler, conf.exp.max_grad_norm, conf.base.verbose, device, rate_schedule[epoch-1])
            else:
                epoch_loss = train_epoch(model, epoch_train_dataloader, criterion, optimizer, scheduler, conf.exp.max_grad_norm, conf.base.verbose, device)
            
            train_finished = time()
            train_elapsed = train_finished - train_start

            epoch_dict = OrderedDict({'epoch': epoch})

            # evaluate every 'eval_interval' epoch
            if epoch % conf.base.eval_interval == 0:
                eval_start = time()
                valid_scores = evaluate_biencoder(model, valid_dataloader, threshold=conf.exp.threshold, device=device)
                eval_finished = time()
                eval_elapsed = eval_finished - eval_start
                
                # log on tensorboard if specified
                if tensorboard is not None:
                    tensorboard.log_metric_from_dict({'train_loss': epoch_loss}, epoch, prefix='Loss')
                    tensorboard.log_metric_from_dict(valid_scores, epoch, prefix='Valid')
                
                epoch_dict.update(valid_scores)
                epoch_dict['train_loss'] = epoch_loss
                epoch_dict['elapsed'] = '%.2f (%.2f + %.2f)' % (train_elapsed + eval_elapsed, train_elapsed, eval_elapsed)
                
                # log epoch info into a file
                file_logger.log_metrics(epoch_dict,epoch)

                # update best scores and parameters
                if valid_scores['accuracy'] > best_em:
                    best_em = valid_scores['accuracy']
                    best_epoch = epoch
                    best_ckpt = os.path.join(log_dir, f'best_ckpt.p')
                    torch.save(model.state_dict(), best_ckpt)
            else:
                if tensorboard is not None:
                    tensorboard.log_metric_from_dict({'train_loss': epoch_loss}, epoch, prefix='Loss')
                
                epoch_dict['train_loss'] = '%.2f' % epoch_loss
                epoch_dict['elapsed'] = '%.2f' % train_elapsed
                file_logger.log_metrics(epoch_dict,epoch, prefix='val_')
            
            print_dict = {k: '%.4f' % v if isinstance(v, float) else v for k, v in epoch_dict.items()}
            print(dict(print_dict))

            # save current model every 'ckpt_interval', disabled if negative
            if conf.base.save_model and conf.base.ckpt_interval > 0 and epoch % conf.base.ckpt_interval == 0:
                torch.save(model.state_dict(), os.path.join(log_dir, f'epoch_{epoch}_ckpt.p'))
            
            print('update negative pairs')
            
            train_dataloader.dataset.update_negative_pairs()
            
        
        print('Restore best model...')
        model.load_state_dict(torch.load(best_ckpt))
        
        print('Evaluate on test set...')
        test_scores = evaluate_biencoder_qr(model, tokenizer, test_dataset, conf.dataset.max_len,beam_size=5, threshold=conf.exp.threshold, lp_alpha=0.2, device=device)

        final_dict = {
            'epoch': 'final test',
            **test_scores
        }

        print(dict(final_dict))
        
        file_logger.log_metrics(final_dict)

        file_logger.save()
        if tensorboard is not None:
            tensorboard.log_metric_from_dict(test_scores, epoch, prefix='Test')
            tensorboard.log_hparams(dict(conf), test_scores)
        
    except KeyboardInterrupt:
        print('[KEYBOARD INTERRUPT] Save log and exit...')
        file_logger.save()