from collections import OrderedDict
import enum

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from itertools import combinations
from utils.metrics import exact_match, batch_masked_cm, precision_recall
import csv

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from IPython import embed

####### conquer_core model #######

# [query index]     [original]      [candidate]     [probability]
# [original]    [candidate]     [probability]


def new_evaluate_tagging_group(model,  vocab, transform, test_dataset, test_dataloader, tokenizer, size, threshold=0.5, device='cpu', log_dir=None):
    model.eval()

    num_test = 0.0
    em = 0
    em1 = 0

    tb_em = []
    acc = []
    f1 = []
    total_num_correct_tokens = 0.0
    total_num_tokens = 0.0


    if log_dir is not None:
        f = open(log_dir+'/test_result.tsv', 'w')
        f.write('qidx\tpred\tlabel\tmask\n')
        f2 = open(log_dir+'/test_result_text.tsv', 'w')
        f2.write('Ori\tTrue\tpred\n')
    

    with torch.no_grad():


        predlabel = []
        recall = []
        precision = []

        wrong_query=[]
        wrong_label=[]
        original_query=[]
        true_reduced=[]
        pred_reduced=[]
        p=0
        r=0

        for batch_id, batch in enumerate(tqdm(test_dataloader, total=len(test_dataloader))):
            original_query=[]
            true_reduced=[]
            pred_reduced=[]
            token_ids = batch['original_ids'].long().to(device)
            reduced_ids = batch['reduced_ids'].long()
            valid_length = batch['original_valid_length']
            segment_ids = batch['original_seg'].long().to(device)
            label = batch['label']
            mask = batch['mask']

            original_query += batch['original']
            true_reduced += batch['reduced']

            model_out = model(token_ids, valid_length, segment_ids)['token_out']
            
            out = model_out
            out = torch.sigmoid(out).cpu()

            for i in range(len(out)):
                original_id = token_ids[i].cpu().numpy()
                out_prob = out[i]
                if transform._tokenizer.__module__.split('.')[0] != 'transformers':
                    original_tokens = transform._tokenizer.convert_ids_to_tokens(original_id)
                    seperate_eojeol=[0]
                    for j in range(len(original_id)):
                        if original_id[j]=='_' or '_' in original_id[j]:
                            seperate_eojeol.append(j)
                    for k in range(len(seperate_eojeol)-1):
                        out_prob[seperate_eojeol[k]+1 : seperate_eojeol[k+1]+1] = max(out_prob[seperate_eojeol[k]+1 : seperate_eojeol[k+1]+1])
                        # out_prob[seperate_eojeol[k]+1 : seperate_eojeol[k+1]+1] = out_prob[seperate_eojeol[k]+1]
                    out[i]=out_prob
                else:
                    original_tokens = transform._tokenizer.convert_ids_to_tokens(original_id)
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
                
            pred = (out > threshold).float().detach().cpu().numpy()


            label = label.numpy()
            mask = mask.numpy()    
            em1 += exact_match(pred, label, mask)
            num_test += len(pred)
            num_correct_tokens = ((pred == label) * mask).sum(1)
            num_tokens = mask.sum(1)

            tb_em.extend(map(str,exact_match(pred, label, mask, reduce=False)))

            for i in range(len(pred)):
                predlabel.append([pred[i],label[i]])
            num_batches = len(pred)

            SEP_ID = transform._tokenizer.convert_tokens_to_ids('[SEP]')
            PAD_ID = transform._tokenizer.convert_tokens_to_ids('[PAD]')

            for i in range(num_batches):
                reduced_id = reduced_ids[i].numpy()
                
                pred_id = pred[i]*token_ids[i].detach().cpu().numpy()
                
                reduced_id = reduced_id[1:valid_length[i]]
                pred_id = pred_id[1:valid_length[i]]

                valid_length_i = valid_length[i]
                valid_label_i = label[i, 1:valid_length_i-1]

                tok, leng, lab, pr = token_ids[i], valid_length[i], label[i], pred[i]
                valid_token = tok[1:leng-1].cpu().numpy()
                if isinstance(vocab, dict):
                    valid_token = [vocab[x] for x in valid_token]  
                elif isinstance(vocab, OrderedDict):
                    valid_token = [tokenizer.ids_to_tokens[x] for x in valid_token]
                else:
                    valid_token = [vocab.idx_to_token[x] for x in valid_token]
                valid_label = lab[1:leng-1]
                pred_label = pr[1:leng-1]
                
                precision.append(precision_score(label[i]*mask[i],pred[i]*mask[i])) 
                recall.append(recall_score(label[i]*mask[i],pred[i]*mask[i]))
                f1.append(f1_score(label[i]*mask[i],pred[i]*mask[i]))
                
                # SEP, PAD 제거
                new_reduced_id = []
                new_pred_id = []
                for idx in range(len(reduced_id)):
                    if reduced_id[idx]!=SEP_ID and reduced_id[idx] !=PAD_ID:
                        new_reduced_id.append(reduced_id[idx])
                    if int(pred_id[idx]) !=SEP_ID and int(pred_id[idx]) != PAD_ID:
                        new_pred_id.append(int(pred_id[idx]))
                
                pred_query = transform._tokenizer.decode(new_pred_id)
                pred_reduced.append(pred_query)
                

                if valid_label.tolist() != pred_label.tolist():
                    
                    reduced_id = transform._tokenizer.convert_ids_to_tokens(reduced_id)
                    pred_id = transform._tokenizer.convert_ids_to_tokens(pred_id)
                    
                    
                    reduced_token = []
                    pred_token = []
                    for idx in range(len(reduced_id)):
                        if reduced_id[idx]!='[SEP]' and reduced_id[idx] !='[PAD]':
                            reduced_token.append(reduced_id[idx])
                        if pred_id[idx]!='[SEP]' and pred_id[idx] != '[PAD]':
                            pred_token.append(pred_id[idx])
                    
                    wrong_label.append([' '.join(reduced_token),' '.join(pred_token)]) 
                valid_answer = [t for t, l in zip(valid_token, valid_label) if l == 1]
                
                total_num_correct_tokens += num_correct_tokens[i]
                total_num_tokens += num_tokens[i]
                acc.append(num_correct_tokens[i]/num_tokens[i])
                

              

                pred_reduce_len = len(np.where(pred_label == 0)[0])


                if log_dir is not None:
                    f.write(f"{batch['query_idx'][i]}\t{' '.join(list(pred[i].astype(int).astype(str)))}\t{' '.join(list(label[i].astype(int).astype(str)))}\t{' '.join(list(mask[i].astype(int).astype(str)))}\n")
                    f2.write(f"{original_query[i]}\t{true_reduced[i]}\t{pred_reduced[i]}\n")

    if log_dir is not None:
        f.close()   
    em = em1/num_test
    acc = sum(acc)/len(acc)
    p = sum(precision)/len(precision)
    r = sum(recall)/len(recall)
    f1 = sum(f1)/len(f1)

    return em ,wrong_query, wrong_label, acc, f1, p, r


def evaluate_ensemble(model1, model2, bert_tokenizer, alpha, test_dataset, max_len, beam_size=5, threshold=0.5, lp_alpha=0.0, device='cpu', log_dir=None):
    labels, preds, masks = [], [], []
    precision, recall, f1, acc ,total_em= [],[],[],[],[]


    num_tokens = 0.0
    num_correct_tokens = 0.0

    print(log_dir)
    if log_dir is not None:
        f = open(log_dir+'/test_result.tsv', 'w')
        f.write('qidx\tpred\tlabel\tmask\n')
        f2 = open(log_dir+'/test_result_text.tsv', 'w')
        f2.write('Ori\tTrue\tpred\n')

    with torch.no_grad():
        for i, x in enumerate(tqdm(test_dataset, total=len(test_dataset))):
            
     
            original = x['original']
            reduced = x['reduced']

            ori_ids, ori_seg, ori_mask = bert_tokenizer(original, max_length=max_len, padding='max_length', return_tensors='np').values()
            token_id = torch.tensor(ori_ids).long().to(device)
            valid_length = torch.tensor(ori_mask.sum())
            valid_length = valid_length.reshape(1)
            segment_id = torch.tensor(ori_seg).long().to(device)
            
            token_out = model1(token_id, valid_length, segment_id )['token_out']

            out = token_out
            out = torch.sigmoid(out).cpu()
            
            
            eojeol=[]    
            original_id = ori_ids.reshape(-1)
            out_prob = out.squeeze(0)
          
            original_tokens = bert_tokenizer.convert_ids_to_tokens(original_id)
            
            temp_prob=[]
            for j, token in enumerate(original_tokens[1:], 1):
                if token == '[SEP]':
                    break
                if token.startswith('##'):
                    temp_prob = max(out_prob[j].item(),eojeol[-1])
                    eojeol[-1] = temp_prob
                    continue
                else:                    
                    eojeol.append(out_prob[j].item())
            
            cur_score = -1
            cur_reduction = original
            
            src_queries = [(cur_reduction, cur_score)] # ('눈썹 문신 후 관리', -1)
            
            all_src_queries = []
            
            stop=False
            step=1
            while not stop:
                if len(cur_reduction.split()) <= 1:
                    break

                # expand candidate
                candidates = expand_candidate_final(src_queries, step)

                
                # calculate pair probabiexpand_candidatelities
                batch_size = 12
                if step == 1:
                    candidate_scores = dict()
                max_batch = np.ceil(len(candidates) / batch_size)
                for b in range(int(max_batch)):
                    batch_start = b*batch_size
                    batch_end = (b+1) * batch_size
                    tmp_scores = score_candidates_final(model2, eojeol, alpha, original, candidates[batch_start:batch_end], bert_tokenizer, step, lp_alpha, device, max_len)
                    candidate_scores.update(tmp_scores)
                

                # extract argmax sample
                sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
                max_candidate, max_candidate_score = sorted_candidates[0]

                # stop if current is the best
                if cur_score >= max_candidate_score:
                    stop = True
 
                else:
                    # update max_candidate
                    cur_score = max_candidate_score
                    cur_reduction = max_candidate
                    all_src_queries += src_queries
                    
                    
                    src_queries = [cand for cand in sorted_candidates[:beam_size] if cand not in all_src_queries] 
                
                step += 1
              
        
            # tokenize and transform best reduction
            original_token_ids, ori_seg, ori_mask = bert_tokenizer(original, max_length=max_len, padding='max_length', return_tensors='np').values()
            original_length = ori_mask.sum()
            reduction_token_ids, red_seg, red_mask = bert_tokenizer(reduced, max_length=max_len, padding='max_length', return_tensors='np').values()
            reduction_length = red_mask.sum()
            pred_token_ids, ped_seg, pred_mask = bert_tokenizer(cur_reduction, max_length=max_len, padding='max_length', return_tensors='np').values()
            pred_length = pred_mask.sum()
            
            # convert best reduction as labels
            original_valid_ids = original_token_ids[0][1:original_length-1]
            reduced_valid_ids = reduction_token_ids[0][1:reduction_length-1]
            pred_valid_ids = pred_token_ids[0][1:pred_length-1]
            
            label = np.zeros((max_len), dtype=np.float)
            _label = np.isin(original_valid_ids, reduced_valid_ids)
            label[1:original_length-1] = _label

            mask = np.zeros((max_len), dtype=np.float)
            mask[1:original_length-1] = 1

            pred = np.zeros((max_len), dtype=np.float)
            _pred = np.isin(original_valid_ids, pred_token_ids)
            pred[1:original_length-1] = _pred

            labels.append(label)
            preds.append(pred)
            masks.append(mask)
    

            precision.append(precision_score(label*mask,pred*mask)) 
            recall.append(recall_score(label*mask,pred*mask))
            f1.append(f1_score(label*mask,pred*mask))
            num_correct_tokens = ((pred == label) * mask).sum()
            num_tokens += mask.sum()
            acc.append(num_correct_tokens/mask.sum())
            em = 1 if num_correct_tokens.sum() == mask.sum() else 0
            total_em.append(em)

            if log_dir is not None:
                f.write(f"{x['query_idx']}\t{' '.join(list(pred.astype(int).astype(str)))}\t{' '.join(list(label.astype(int).astype(str)))}\t{' '.join(list(mask.astype(int).astype(str)))}\n")
                f2.write(f"{original}\t{reduced}\t{cur_reduction}\n")


    labels = np.stack(labels)
    preds = np.stack(preds)
    masks = np.stack(masks)

    score = {}
    score['Acc.'] = sum(acc)/len(acc)
    score['precision.']= sum(precision)/len(precision)
    score['recall.'] = sum(recall)/len(recall) 
    score['f1.'] = sum(f1)/len(f1)
    score['EM'] = sum(total_em)/len(total_em)
    
    if log_dir is not None:
        f.close()
    
    return score


def expand_candidate_final(src_queries, step):
    candidates = []
   
    for query, _ in src_queries:
        query_tokens = query.split()

        
        for i in range(len(query_tokens)):
            cand_tokens = [tok for j, tok in enumerate(query_tokens) if i != j]
            candidates.append(' '.join(cand_tokens))
         
    return candidates

def score_candidates_final(model2, out, alpha,original, candidates, bert_tokenizer, step, lp_alpha, device, max_len):
    token_ids = []
    valid_length = []
    segment_ids = []
    tb_score = []

    for cand in candidates:
        tok_id, seg_ids, ori_mask = bert_tokenizer(original, cand, max_length=max_len, padding='max_length', return_tensors='np').values()
        valid_len = ori_mask.sum()
        token_ids.append(tok_id[0])
        valid_length.append(valid_len)
        segment_ids.append(seg_ids[0])
        tb=0
        tb_idx = np.where(np.isin(original.split(),cand.split()) == True)[0].tolist()
        
        for i in range(len(original.split())):
            
            if i in tb_idx:
                tb+=out[i]
            else:
                tb+=(1-out[i])

        tb=tb/len((original.split()))

        tb_score.append((1-alpha)*tb)
        
        
    
    token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)
    valid_length = torch.tensor(np.stack(valid_length), dtype=torch.int).to(device)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(device)

    model_out = model2(token_ids, valid_length, segment_ids)

    cls_out = torch.sigmoid(model_out['cls_out']).detach().cpu().numpy()
    cls_out = alpha*cls_out + tb_score
    
    if lp_alpha > 0:
        cur_lp = lp(step, lp_alpha)
    else:
        cur_lp = 1.0

     #* cur_lp
    return {c: cls_out[i] for i, c in enumerate(candidates)}


## pair-based candidate length penalty ##

def lp(step, alpha=0.2):
    return (5 + 1) ** alpha / (5 + step) ** alpha



#################### conquer_sub ####################

def evaluate_biencoder(model, test_dataloader, threshold=0.5, pooling=False, device='cpu') -> dict:
    model.eval()
    
    num_correct = 0.0
    num_test = 0.0
    score = {
        'num_correct': 0.0,
        'num_test': 0.0
    }
    with torch.no_grad():
        
        for batch_id, batch in enumerate(tqdm(test_dataloader, total=len(test_dataloader))):
            
            ori_token_ids = batch['original_tok'].long().to(device)
            ori_valid_length = batch['original_valid_length']
            ori_segment_ids = batch['original_segment_ids'].long().to(device)
            cand_token_ids = batch['reduced_tok'].long().to(device)
            cand_valid_length = batch['reduced_valid_length']
            cand_segment_ids = batch['reduced_segment_ids'].long().to(device)
            label = batch['label']

            model_out = model(ori_token_ids, ori_valid_length, ori_segment_ids)
            model_out2 = model(cand_token_ids, cand_valid_length, cand_segment_ids)
            # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            
            out = torch.sigmoid(torch.diag(model_out['cls_out'] @ model_out2['cls_out'].T))
            # out = torch.sigmoid(torch.sum(model_out['cls_out']* model_out2['cls_out'], dim=1))
            
            
            pred = (out > threshold).float().detach().cpu().numpy()
            label = label.numpy()
            correct = pred == label
            
            score['num_correct'] +=  correct.astype('int').sum()
            score['num_test'] += len(correct)

        score['accuracy'] = (score['num_correct'] / score['num_test'])
    return score




def evaluate_biencoder_qr(model, bert_tokenizer, test_dataset, max_len, beam_size=5, threshold=0.5, lp_alpha=0.0, pooling=False, device='cpu', log_dir=None):
    labels, preds, masks = [], [], []
    precision, recall, f1, acc = [],[],[], []
    golds, reductions, correct, maxcand, wrong, result = [], [], [], [], [], []
    num1 , num2, num3, num4, num5 = 0, 0, 0, 0, 0
    same, different = 0,0

    num_tokens = 0.0
    num_correct_tokens = 0.0

    
    if log_dir is not None:
        f = open(log_dir+'/test_result.tsv', 'w')
        f.write('qidx\tpred\tlabel\tmask\n')
        f2 = open(log_dir+'/test_result_text.tsv', 'w')
        f2.write('Ori\tTrue\tpred\n')
    for i, x in enumerate(tqdm(test_dataset, total=len(test_dataset))):
        
        original = x['original']
        reduced = x['reduced']


        cur_score = -1
        cur_reduction = original
        
        src_queries = [(cur_reduction, cur_score)] # ('눈썹 문신 후 관리', -1)
        
        all_src_queries = []
        
        stop=False
        step=1
        while not stop:
            if len(cur_reduction.split()) <= 1:
                break

            # expand candidate
            candidates = expand_candidate(src_queries, step)
   
            batch_size = 12
            if step == 1:
                candidate_scores = dict()
            max_batch = np.ceil(len(candidates) / batch_size)
            for b in range(int(max_batch)):
                batch_start = b*batch_size
                batch_end = (b+1) * batch_size
                tmp_scores = bi_score_candidates(model, original, candidates[batch_start:batch_end], bert_tokenizer, step, lp_alpha, pooling, device, max_len)
                candidate_scores.update(tmp_scores)
            
           

            
            
            # extract argmax sample
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
            max_candidate, max_candidate_score = sorted_candidates[0]

            # stop if current is the best
            if cur_score >= max_candidate_score:
                stop = True

            else:
                # update max_candidate
                cur_score = max_candidate_score
                cur_reduction = max_candidate
                all_src_queries += src_queries
                src_queries = [cand for cand in sorted_candidates[:beam_size] if cand not in all_src_queries]
        
            step += 1

    
        # tokenize and transform best reduction
        original_token_ids, ori_seg, ori_mask = bert_tokenizer(original, max_length=max_len, padding='max_length', return_tensors='np').values()
        original_length = ori_mask.sum()
        reduction_token_ids, red_seg, red_mask = bert_tokenizer(reduced, max_length=max_len, padding='max_length', return_tensors='np').values()
        reduction_length = red_mask.sum()
        pred_token_ids, ped_seg, pred_mask = bert_tokenizer(cur_reduction, max_length=max_len, padding='max_length', return_tensors='np').values()
        pred_length = pred_mask.sum()

        # convert best reduction as labels
        original_valid_ids = original_token_ids[0][1:original_length-1]
        reduced_valid_ids = reduction_token_ids[0][1:reduction_length-1]
        pred_valid_ids = pred_token_ids[0][1:pred_length-1]

        label = np.zeros((max_len), dtype=np.float)
        _label = np.isin(original_valid_ids, reduced_valid_ids)
        label[1:original_length-1] = _label

        mask = np.zeros((max_len), dtype=np.float)
        mask[1:original_length-1] = 1

        pred = np.zeros((max_len), dtype=np.float)
        _pred = np.isin(original_valid_ids, pred_valid_ids)
        pred[1:original_length-1] = _pred

        labels.append(label)
        preds.append(pred)
        masks.append(mask)

        precision.append(precision_score(label*mask,pred*mask)) 
        recall.append(recall_score(label*mask,pred*mask))
        f1.append(f1_score(label*mask,pred*mask))
        num_correct_tokens = ((pred == label) * mask).sum()
        num_tokens += mask.sum()
        
        acc.append(num_correct_tokens/mask.sum())

        golds.append(reduced)
        maxcand.append(cur_reduction)
        diff = len(reduced.split())-len(cur_reduction.split())

        reductions.append(src_queries)

        result.append([original, reduced, cur_reduction])
        if reduced == cur_reduction:
            correct.append([original, cur_reduction])
        
            
        if reduced == cur_reduction and original == cur_reduction:
            num1+=1
        elif original == reduced and reduced != cur_reduction:
            num2+=1
            wrong.append([original, reduced, cur_reduction])
        elif original != reduced and original == cur_reduction:
            num3+=1
        elif original != reduced and original != cur_reduction and reduced != cur_reduction:
            num4+=1
        elif original != reduced and reduced == cur_reduction:
            num5+=1
        

        if original == reduced:
            same+=1
        else:
            different+=1
        if log_dir is not None:
            f.write(f"{x['query_idx']}\t{' '.join(list(pred.astype(int).astype(str)))}\t{' '.join(list(label.astype(int).astype(str)))}\t{' '.join(list(mask.astype(int).astype(str)))}\n")
            f2.write(f"{original}\t{reduced}\t{cur_reduction}\n")

    labels = np.stack(labels)
    preds = np.stack(preds)
    masks = np.stack(masks)

    score = {}
    score['Acc.'] = sum(acc)/len(acc)
    score['precision.']= sum(precision)/len(precision)
    score['recall.'] = sum(recall)/len(recall) 
    score['f1.'] = sum(f1)/len(f1)

    # exact match
    score['EM'] = exact_match(preds, labels, masks) / len(test_dataset)
    pb_em = []
    pb_em.extend(map(str,exact_match(preds, labels, masks, reduce=False)))
    
  
    if log_dir is not None:
        f.close()
    
    return score


def expand_candidate(src_queries, step):
    candidates = []

    for query, _ in src_queries:
        query_tokens = query.split()
     
        for i in range(len(query_tokens)):
            cand_tokens = [tok for j, tok in enumerate(query_tokens) if i != j]
            candidates.append(' '.join(cand_tokens))
         
    return candidates

def bi_score_candidates(model, original, candidates, transform, step, lp_alpha, pooling, device, max_len):
    ori_token_ids = []
    ori_valid_length = []
    ori_segment_ids = []
    cand_token_ids = []
    cand_valid_length = []
    cand_segment_ids = []

    for cand in candidates:

        ori_tok_id, ori_seg_ids, ori_attention_mask = transform(original, max_length=max_len, padding='max_length', return_tensors='np').values()
        ori_valid_len = ori_attention_mask.sum()
        # from IPython import embed; embed()
        ori_tok_id, ori_seg_ids = ori_tok_id[0], ori_seg_ids[0]
        
        cand_tok_id, cand_seg_ids, cand_attention_mask = transform(cand, max_length=max_len, padding='max_length', return_tensors='np').values()
        cand_valid_len = cand_attention_mask.sum()
        cand_tok_id, cand_seg_ids = cand_tok_id[0], cand_seg_ids[0]

        ori_token_ids.append(ori_tok_id)
        ori_valid_length.append(ori_valid_len)
        ori_segment_ids.append(ori_seg_ids)
        
        cand_token_ids.append(cand_tok_id)
        cand_valid_length.append(cand_valid_len)
        cand_segment_ids.append(cand_seg_ids)
    
    ori_token_ids = torch.tensor(ori_token_ids, dtype=torch.long).to(device)
    ori_valid_length = torch.tensor(np.stack(ori_valid_length), dtype=torch.int).to(device)
    ori_segment_ids = torch.tensor(ori_segment_ids, dtype=torch.long).to(device)
    
    cand_token_ids = torch.tensor(cand_token_ids, dtype=torch.long).to(device)
    cand_valid_length = torch.tensor(np.stack(cand_valid_length), dtype=torch.int).to(device)
    cand_segment_ids = torch.tensor(cand_segment_ids, dtype=torch.long).to(device)
    model_out = model(ori_token_ids, ori_valid_length, ori_segment_ids)
    model_out2 = model(cand_token_ids, cand_valid_length, cand_segment_ids)
    

    output = torch.diag(model_out['cls_out'] @ model_out2['cls_out'].T)

    cls_out = output.detach().cpu().numpy()
    
    
    if lp_alpha > 0:
        cur_lp = lp(step, lp_alpha)
    else:
        cur_lp = 1.0

     #* cur_lp
    return {c: cls_out[i] for i, c in enumerate(candidates)}


def eval_by_text_file(txt_file):
    file_new = open(txt_file, 'r')
    lines = file_new.readlines()
    if 'tsv' in txt_file:
        delimiter = '\t'
    elif 'csv' in txt_file:
        delimiter = ','
    
    em_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    acc_list = []

    em_list_single = []
    precision_list_single = []
    recall_list_single = []
    f1_list_single = []
    acc_list_single = []

    em_list_multi = []
    precision_list_multi = []
    recall_list_multi = []
    f1_list_multi = []
    acc_list_multi = []

    num_test = 0

    for i, line in enumerate(tqdm(lines, desc='evaluating...', dynamic_ncols=True)):
        if i == 0:
            continue
        line = line.strip().split(delimiter)
        original_query = line[0]
        true_reduced = line[1]
        try:
            pred_reduced = line[2]
        except:
            pred_reduced = ''

        original_words = original_query.split()
        reduced_words = true_reduced.split()
        generated_words = pred_reduced.split()


        # em
        em_list.append(int(generated_words == reduced_words))

        # precision, recall, f1
        label = np.zeros(len(original_words))
        label = np.isin(original_words, reduced_words)

        pred = np.zeros(len(original_query))
        pred = np.isin(original_words, generated_words)

        precision_list.append(precision_score(label,pred, zero_division=0)) 
        recall_list.append(recall_score(label,pred))
        f1_list.append(f1_score(label,pred))
        
        # acc
        acc_list.append((label == pred).mean())

        num_remove_word = len(original_words) - len(reduced_words)
        if num_remove_word == 1:
            em_list_single.append(em_list[-1])
            precision_list_single.append(precision_list[-1])
            recall_list_single.append(recall_list[-1])
            f1_list_single.append(f1_list[-1])
            acc_list_single.append(acc_list[-1])
        
        else:
            em_list_multi.append(em_list[-1])
            precision_list_multi.append(precision_list[-1])
            recall_list_multi.append(recall_list[-1])
            f1_list_multi.append(f1_list[-1])
            acc_list_multi.append(acc_list[-1])
        
        num_test += 1

    em = sum(em_list)/num_test
    acc = sum(acc_list)/num_test
    p = sum(precision_list)/num_test
    r = sum(recall_list)/num_test
    f1 = sum(f1_list)/num_test

    print(f'accuracy: {acc:.3f} ({sum(acc_list)}/{num_test})')
    print(f'precision: {p:.3f} ({sum(precision_list)}/{num_test})')
    print(f'recall: {r:.3f} ({sum(recall_list)}/{num_test})')
    print(f'f1_score: {f1:.3f} ({sum(f1_list)}/{num_test})')
    print(f'em: {em:.3f} ({sum(em_list)}/{num_test})')
    print()
    print(f'accuracy_single: {sum(acc_list_single)/len(acc_list_single):.3f} ({sum(acc_list_single)}/{len(acc_list_single)})')
    print(f'precision_single: {sum(precision_list_single)/len(precision_list_single):.3f} ({sum(precision_list_single)}/{len(precision_list_single)})')
    print(f'recall_single: {sum(recall_list_single)/len(recall_list_single):.3f} ({sum(recall_list_single)}/{len(recall_list_single)})')
    print(f'f1_score_single: {sum(f1_list_single)/len(f1_list_single):.3f} ({sum(f1_list_single)}/{len(f1_list_single)})')
    print(f'em_single: {sum(em_list_single)/len(em_list_single):.3f} ({sum(em_list_single)}/{len(em_list_single)})')
    print()
    print(f'accuracy_multi: {sum(acc_list_multi)/len(acc_list_multi):.3f} ({sum(acc_list_multi)}/{len(acc_list_multi)})')
    print(f'precision_multi: {sum(precision_list_multi)/len(precision_list_multi):.3f} ({sum(precision_list_multi)}/{len(precision_list_multi)})')
    print(f'recall_multi: {sum(recall_list_multi)/len(recall_list_multi):.3f} ({sum(recall_list_multi)}/{len(recall_list_multi)})')
    print(f'f1_score_multi: {sum(f1_list_multi)/len(f1_list_multi):.3f} ({sum(f1_list_multi)}/{len(f1_list_multi)})')
    print(f'em_multi: {sum(em_list_multi)/len(em_list_multi):.3f} ({sum(em_list_multi)}/{len(em_list_multi)})')

    return em, acc, f1, p, r