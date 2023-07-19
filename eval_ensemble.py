######## IMPORTS
import os
import argparse
import torch
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score
from collections import defaultdict


from models import model_builder
from config import load_config_from_path


from utils.datasets import load_datasets
from utils.evaluation import evaluate_ensemble

def main(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device = torch.device("cuda"if torch.cuda.is_available() else "cpu")

    log_dir1 = os.path.dirname(args.ckpt_file1)
    log_dir2 = os.path.dirname(args.ckpt_file2)


    conf1 = load_config_from_path(os.path.join(log_dir1, 'hparams.yaml'))
    conf2 = load_config_from_path(os.path.join(log_dir2, 'hparams.yaml'))

    #####################################################
    #                   Load BERT                       #
    #####################################################
    print('Load BERT model...')
    
    print(f'> You are now using HuggingFace library.')
    from transformers import AutoTokenizer, AutoModel
    bertmodel1 = AutoModel.from_pretrained(conf1.model.bert_model)
    tokenizer1 = AutoTokenizer.from_pretrained(conf1.model.bert_model)
    bertmodel2 = AutoModel.from_pretrained(conf2.model.bert_model)
    tokenizer2 = AutoTokenizer.from_pretrained(conf2.model.bert_model)

    

    print('Build BERT-Reduction model...')
    model1_name = 'bert'
    model1 = model_builder(model1_name, bert=bertmodel1, device=device, **conf1.model).to(device)
    model1.load_state_dict(torch.load(args.ckpt_file1))
    model1.eval()

    print('Build sentence pair classification model...')
    model2_name = 'bert_pair'
    model2 = model_builder(model2_name, bert=bertmodel2, device=device, **conf2.model).to(device)
    model2.load_state_dict(torch.load(args.ckpt_file2))
    model2.eval()
        
    print('Load dataset...')
    
    _, _, test_dataset = load_datasets(
        testfile=args.test_file,
        bert_tokenizer=tokenizer1, 
        ori_idx=0,
        reduced_idx=1,  
        **conf2.dataset)
    score = evaluate_ensemble(model1, model2, tokenizer1, args.alpha, test_dataset, conf2.dataset.max_len, beam_size=args.beam_size, threshold=0.5, lp_alpha=args.lp_alpha, device=device,log_dir=log_dir2)
    
    score_print = {k: '%.4f' % v for k, v in score.items()}
    print(score_print)

    args.by_reduced_term=True
    if args.by_query_length or args.by_reduced_term:
        fname = 'test_by_query_length.txt' if args.by_query_length else 'test_by_reduced_term.txt'
        with open(os.path.join(log_dir2, fname), 'w', encoding='utf-8') as nf:
            nf.write(f"category #samples ACC F1 EM\t\n")

        
            f = open(os.path.join(log_dir2, 'test_result.tsv'), 'r')
            tmp_dict = defaultdict(dict) # pred, label, mask, em
            accuracy=[]
            total_em, total_precision,total_recall,total_acc,total_f1=[],[],[],[],[]
            for line in f:
                qidx, pred, label, mask = line.strip().split('\t')
                if qidx == 'qidx': continue
                
                pred = np.array([int(n) for n in pred.split()])
                label = np.array([int(n) for n in label.split()])
                mask = np.array([int(n) for n in mask.split()])
                
                token_correct = (pred == label) * mask
                em = 1 if token_correct.sum() == mask.sum() else 0
                total_em.append(em)
                accuracy.append(token_correct.sum()/mask.sum())
                if args.by_query_length:
                    qlen = len(test_dataset[int(qidx)]['original'].split())
                    qtype=qlen
                else:
                    qlen = len([w for w in test_dataset[int(qidx)]['original'].split() if w not in test_dataset[int(qidx)]['reduced'].split()])
                    qtype=qlen
                
                if qtype not in tmp_dict:
                    tmp_dict[qtype]['pred'] = [pred]
                    tmp_dict[qtype]['label'] = [label]
                    tmp_dict[qtype]['mask'] = [mask]
                    tmp_dict[qtype]['em'] = [em]
                else:
                    tmp_dict[qtype]['pred'].append(pred)
                    tmp_dict[qtype]['label'].append(label)
                    tmp_dict[qtype]['mask'].append(mask)
                    tmp_dict[qtype]['em'].append(em)
            f.close()
            print(f'accuracy: {sum(accuracy)/len(accuracy)}')
            for k in sorted(tmp_dict.keys()):
                v = tmp_dict[k]
                
                preds = np.concatenate(v['pred'])
                labels = np.concatenate(v['label'])
                masks = np.concatenate(v['mask'])
                ems = np.array(v['em'])
                precision=[]
                recall=[]
                f1=[]
                acc=[]
                
                for i in range(len(v['pred'])):
                    precision.append(precision_score(v['label'][i]*v['mask'][i],v['pred'][i]*v['mask'][i])) 
                    recall.append(recall_score(v['label'][i]*v['mask'][i],v['pred'][i]*v['mask'][i]))
                    f1.append(f1_score(v['label'][i]*v['mask'][i],v['pred'][i]*v['mask'][i]))
                    acc.append(((v['pred'][i]==v['label'][i])*v['mask'][i]).sum() / v['mask'][i].sum())
                    total_precision.append(precision_score(v['label'][i]*v['mask'][i],v['pred'][i]*v['mask'][i])) 
                    total_recall.append(recall_score(v['label'][i]*v['mask'][i],v['pred'][i]*v['mask'][i]))
                    total_f1.append(f1_score(v['label'][i]*v['mask'][i],v['pred'][i]*v['mask'][i]))
                    total_acc.append(((v['pred'][i]==v['label'][i])*v['mask'][i]).sum() / v['mask'][i].sum())

                nf.write(f'{k} {ems.shape[0]} {sum(acc)/len(acc):.4f} {sum(precision)/len(precision):.4f} {sum(recall)/len(recall):.4f} {sum(f1)/len(f1):.4f} {ems.mean():.4f}\n')
            nf.write(f'{k} {ems.shape[0]} {sum(total_acc)/len(total_acc):.4f} {sum(total_precision)/len(total_precision):.4f} {sum(total_recall)/len(total_recall):.4f} {sum(total_f1)/len(total_f1):.4f} {sum(total_em)/len(total_em):.4f}\n')

        print(f">>> {fname} saved in log_dir")
                

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_file1', type=str, default='saves/sample/best_ckpt.p')
    parser.add_argument('--ckpt_file2', type=str, default='saves/sample/best_ckpt.p')
    parser.add_argument('--test_file', type=str, default='./data/sample/test.tsv')
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--lp_alpha', type=float, default=0.2)
    parser.add_argument('--gen_reduction', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--by_query_length', action='store_true')
    parser.add_argument('--by_reduced_term', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())