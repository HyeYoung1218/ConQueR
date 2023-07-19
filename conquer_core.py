######## IMPORTS
import os
import sys

from time import time

import torch
import torch.nn as nn
from IPython import embed


# import gluonnlp as nlp
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from logger import FileLogger, TensorboardLogger
from utils.metrics import exact_match
from utils.functions import set_random_seed
from utils.datasets import load_datasets


from models import model_builder

from config import load_config, update_params

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
from transformers import AutoTokenizer, AutoModel, AutoConfig
bertmodel = AutoModel.from_pretrained(conf.model.bert_model)
tokenizer = AutoTokenizer.from_pretrained(conf.model.bert_model)

#####################################################
#                   Load dataset                    #
#####################################################


train_file = os.path.join(conf.dataset.datadir, 'sample/train.tsv')
valid_file = os.path.join(conf.dataset.datadir, 'sample/valid.tsv')
test_file = os.path.join(conf.dataset.datadir, 'sample/test.tsv')

print('Load dataset...')

train_dataset, valid_dataset, test_dataset = load_datasets(
                                            trainfile=train_file,
                                            validfile=valid_file,
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
print('Build BERT-Reduction model...')
model_name = 'bert_tree' if conf.model.tree_transformer else 'bert'
model = model_builder(model_name, bert=bertmodel, device=device, **conf.model).to(device)
model.conf = conf

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
criterion = nn.BCEWithLogitsLoss(reduction='none')
criterion_core = nn.MSELoss() if conf.model.pred_num_core else None

t_total = len(train_dataloader) * conf.exp.num_epochs
warmup_step = int(t_total * conf.exp.warmup_ratio)
print(f"truncated_loss = {conf.model.truncated_loss}")
if conf.model.truncated_loss:
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

#### File logger
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




def train_epoch(model, train_dataloader, criterion, optimizer, scheduler, max_grad_norm=0.0, verbose=0,
                pred_num_core=False, criterion_core=None, contrastive=False, augment_ratio=0.0, augment_lambda=0.1, device='cpu', truncated_loss=False, forget_rate=0.1):
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
        token_ids = batch['original_ids'].long().to(device)
        valid_length = batch['original_valid_length'].to(device)
        segment_ids = batch['original_seg'].long().to(device)
        label = batch['label'].to(device)
        mask = batch['mask'].to(device)

        model_out = model(token_ids, valid_length, segment_ids)
        token_out = model_out['token_out']
       

        _token_loss = criterion(token_out, label)
        
        if truncated_loss:
            token_loss = (_token_loss * mask).sum(axis=1)
            _token_loss_mean = token_loss/(valid_length.to(device)-2)
            
            idx_token_loss_sorted = np.argsort(_token_loss_mean.data.cpu()).cuda()
            _token_loss_sorted = _token_loss_mean[idx_token_loss_sorted]

            remember_rate = 1 - forget_rate
            num_remember = int(remember_rate * len(_token_loss_sorted))
            import pickle
            
            idx_token_update = idx_token_loss_sorted[:num_remember]
            with open('idf_token_update_epoch0.pkl', 'wb') as f:
                pickle.dump(idx_token_update, f, protocol=4)
            loss_update = criterion(token_out[idx_token_update], label[idx_token_update])
            

            
            if augment_ratio > 0:
                augment_mask = batch['augment_mask'].to(device)
                masked_token_loss = (loss_update * mask[idx_token_update]).sum(1)
                loss_update = (masked_token_loss * (1 - augment_mask) + masked_token_loss * augment_mask * augment_lambda).mean()
            else:
                loss_update = (loss_update * mask[idx_token_update]).sum(1).mean()
            
            if pred_num_core:
                num_core_out = model_out['num_core_out']
                num_core = label.sum(1).float()
                num_remove = valid_length-num_core
                core_loss = criterion_core(num_core_out, num_remove)
                loss = loss_update + 0.1*core_loss
            else:
                loss = loss_update

            if contrastive:
                cont_temp = model.conf.model.cont_temp # 0.01, 0.1, 1, 10
                cont_lambda = model.conf.model.cont_lambda 
                cos_sim = model_out['cos_sim'] # (B, 1, L)
                neg_cos = cos_sim.masked_fill(mask.bool().unsqueeze(1) != True, -1e9) # (B, 1, L) # mask out padding
                neg_cos = neg_cos.masked_fill(label.bool().unsqueeze(1) == True, -1e9) # (B, 1, L) # mask out positive
                pos_cos = cos_sim.masked_fill(label.bool().unsqueeze(1) != True, -1e9) # (B, 1, L) # mask out negative
                cons_pos = torch.exp(pos_cos/ cont_temp ) # (B, 1, L)
                cons_neg = torch.sum(torch.exp(neg_cos / cont_temp ), dim=2) # (B, 1)
                cons_div = cons_pos / (cons_neg.unsqueeze(-1) + cons_pos) # (B, 1, L)
                cons_div = cons_div.masked_fill(mask.bool().unsqueeze(1) != True, 1) # (B, 1, L) # mask out padding
                cons_div = cons_div.masked_fill(label.bool().unsqueeze(1) != True, 1) # (B, 1, L) # mask out negative
                # loss_contrastive = -torch.log(cons_div).mean() 
                loss_contrastive = -torch.log(cons_div).squeeze(1).sum(1).mean()
                loss = loss + cont_lambda * loss_contrastive

            

            
        
        else:
            if augment_ratio > 0:
                augment_mask = batch['augment_mask'].to(device)
                masked_token_loss = (_token_loss * mask).sum(1)
                token_loss = (masked_token_loss * (1 - augment_mask) + masked_token_loss * augment_mask * augment_lambda).mean()
            else:
                token_loss = (_token_loss * mask).sum(1).mean()
            
            if pred_num_core:
                num_core_out = model_out['num_core_out']
                num_core = label.sum(1).float()
                num_remove = valid_length-num_core
                core_loss = criterion_core(num_core_out, num_core)

                loss = token_loss + core_loss
            else:
                loss = token_loss
        
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


def evaluate(model, test_dataloader, threshold=0.5):
    model.eval()
    
    num_test = 0.0
    em = 0.0
    confusion_matrix = np.zeros((2, 2))
    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(test_dataloader, total=len(test_dataloader))):
            token_ids = batch['original_ids'].long().to(device)
            valid_length = batch['original_valid_length']
            segment_ids = batch['original_seg'].long().to(device)
            label = batch['label']
            mask = batch['mask']

            model_out = model(token_ids, valid_length, segment_ids)
            out = torch.sigmoid(model_out['token_out'])

            for i in range(len(out)):
                original_id = token_ids[i].cpu().numpy()
                out_prob = out[i]

                original_tokens = tokenizer.convert_ids_to_tokens(original_id)
                token_start_idx = 0
                for j, token in enumerate(original_tokens[1:], 1):
                    if token == '[SEP]':
                        break
                    if token.startswith('##'):
                        continue
                    else:
                        if token_start_idx > 0:
                            out_prob[token_start_idx:j] = max(out_prob[token_start_idx: j])
                        token_start_idx = j
                out[i] = out_prob

            # Token acc
            pred = (out > threshold).detach().float().cpu().numpy()
            label = label.numpy()

            mask = mask.numpy()
            
            em += exact_match(pred, label, mask)

            num_test += len(pred)

        ret = OrderedDict({'EM': em/num_test})

    return ret

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
            if conf.dataset.augment_ratio > 0:
                train_dataloader.dataset.augment_data(conf.dataset.augment_type, conf.dataset.augment_ratio, conf.dataset.num_truncate)
             
            epoch_train_dataloader = train_dataloader

            # Training phase
            train_start = time()
            if conf.model.truncated_loss:
                epoch_loss = train_epoch(model, epoch_train_dataloader, criterion, optimizer, scheduler, 
                    conf.exp.max_grad_norm, conf.base.verbose, conf.model.pred_num_core, criterion_core, conf.model.contrastive,
                    conf.dataset.augment_ratio, conf.dataset.augment_lambda, device, conf.model.truncated_loss, rate_schedule[epoch-1])
            else:
                epoch_loss = train_epoch(model, epoch_train_dataloader, criterion, optimizer, scheduler, 
                    conf.exp.max_grad_norm, conf.base.verbose, conf.model.pred_num_core, criterion_core, conf.model.contrastive,
                    conf.dataset.augment_ratio, conf.dataset.augment_lambda, device)

            train_finished = time()
            train_elapsed = train_finished - train_start

            epoch_dict = OrderedDict({'epoch': epoch})

            if epoch % conf.base.eval_interval == 0:
                eval_start = time()
                valid_scores = evaluate(model, valid_dataloader, threshold=conf.exp.threshold)
                eval_finished = time()
                eval_elapsed = eval_finished - eval_start
                
                if tensorboard is not None:
                    tensorboard.log_metric_from_dict({'train_loss': epoch_loss}, epoch, prefix='Loss')
                    tensorboard.log_metric_from_dict(valid_scores, epoch, prefix='Valid')
                
                epoch_dict.update(valid_scores)
                epoch_dict['train_loss'] = epoch_loss
                epoch_dict['elapsed'] = '%.2f (%.2f + %.2f)' % (train_elapsed + eval_elapsed, train_elapsed, eval_elapsed)
                file_logger.log_metrics(epoch_dict,epoch)

                if valid_scores['EM'] > best_em:
                    best_em = valid_scores['EM']
                    best_epoch = epoch
                    best_ckpt = os.path.join(log_dir, f'best_ckpt.p')
                    torch.save(model.state_dict(), best_ckpt)
            else:
                if tensorboard is not None:
                    tensorboard.log_metric_from_dict({'train_loss': epoch_loss}, epoch, prefix='Loss')
                epoch_dict['train_loss'] = '%.2f' % epoch_loss
                epoch_dict['elapsed'] = '%.2f' % train_elapsed
                file_logger.log_metrics(epoch_dict, prefix='val_')
            
            print_dict = {k: '%.4f' % v if isinstance(v, float) else v for k, v in epoch_dict.items()}
            print(dict(print_dict))
            if conf.base.save_model and epoch % conf.base.ckpt_interval == 0:
                torch.save(model.state_dict(), os.path.join(log_dir, f'epoch_{epoch}_ckpt.p'))
            


        print('Restore best model...')
        model.load_state_dict(torch.load(best_ckpt))
        
        print('Evaluate on test set...')
        test_scores = evaluate(model, test_dataloader, threshold=conf.exp.threshold)

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
