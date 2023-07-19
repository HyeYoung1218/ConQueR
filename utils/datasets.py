import os
import torch
import random
import pickle
import itertools
import numpy as np
import gluonnlp as nlp
from tqdm import tqdm
from torch.utils.data import Dataset
import random
import csv
# tagging-based


def load_datasets(trainfile=None, validfile=None, testfile=None, **args):
    train_dataset = ReductionDataset(datapath=trainfile, **args) if trainfile is not None else None
    valid_dataset = ReductionDataset(datapath=validfile, **args) if validfile is not None else None
    test_dataset = ReductionDataset(datapath=testfile, **args) if testfile is not None else None
    return train_dataset, valid_dataset, test_dataset


# pair-based(point-wise)
def load_pair_datasets(trainfile=None, validfile=None, testfile=None, **args):
    train_dataset = SentencePairDataset(datapath=trainfile, is_test=False, **args) if trainfile is not None else None
    valid_dataset = SentencePairDataset(datapath=validfile, is_test=True, **args) if validfile is not None else None
    test_dataset = SentencePairDataset(datapath=testfile, is_test=True, **args) if testfile is not None else None
    return train_dataset, valid_dataset, test_dataset
# final valid,test

# pair-based(pair-wise only)
def load_pair_only_datasets(trainfile=None, validfile=None, testfile=None, **args):
    train_dataset = SentencePairLossOnlyDataset(datapath=trainfile, is_test=False, **args) if trainfile is not None else None
    valid_dataset = SentencePairLossOnlyDataset(datapath=validfile, is_test=True, **args) if validfile is not None else None
    test_dataset = SentencePairLossOnlyDataset(datapath=testfile, is_test=True, **args) if testfile is not None else None
    return train_dataset, valid_dataset, test_dataset


def load_biencodier_pairwise_datasets(trainfile=None, validfile=None, testfile=None, **args):
    train_dataset = BiencoderPairLossDataset(datapath=trainfile, is_test=False, **args) if trainfile is not None else None
    valid_dataset = BiencoderPairLossDataset(datapath=validfile, is_test=True, **args) if validfile is not None else None
    test_dataset = BiencoderPairLossDataset(datapath=testfile, is_test=True, **args) if testfile is not None else None
    return train_dataset, valid_dataset, test_dataset

def load_generate_datasets(trainfile=None, validfile=None, testfile=None, **args):
    train_dataset = GenerateDataset(datapath=trainfile, is_test=False, **args) if trainfile is not None else None
    valid_dataset = GenerateDataset(datapath=validfile, is_test=True, **args) if validfile is not None else None
    test_dataset = GenerateDataset(datapath=testfile, is_test=True, **args) if testfile is not None else None
    return train_dataset, valid_dataset, test_dataset

####### Tagging-based ############



class ReductionDataset(Dataset):
    def __init__(self, datapath, bert_tokenizer, ori_idx=0, reduced_idx=1, max_len=32, phr_sep=False, phr_sep_token='[SEP]', debug=False, **args):
        self.datapath = datapath
        self.ori_idx = ori_idx
        self.reduced_idx = reduced_idx
        self.max_len = max_len
        self.bert_tokenizer = bert_tokenizer
        self.phr_sep = phr_sep
        self.phr_sep_token = phr_sep_token
        self.phr_sep_token_idx = bert_tokenizer.vocab[self.phr_sep_token]
        self.debug = debug
        self._base_data = self.setup()
        self._data = self._base_data

    """
    - original: str, original query
    - original ids: np.array, BERT input form of original query
    - original ids len: valid length of original ids
    - reduced: str, reduced query
    - reduced ids: np.array, BERT input form of reduced query
    - reduced ids len: valid length of reduced ids
    - label: np.array, binary array indicating core (1) or reduce (0)
    - mask: np.array, binary array indicating whether to calculate loss (1) or not (0)
    - num_reduced: int, # of reduced token
    """

    def setup(self):
        processed = []
        idx = 0


        dataset = nlp.data.TSVDataset(self.datapath, field_indices=[self.ori_idx, self.reduced_idx], num_discard_samples=0)
        
        
        for i,x in enumerate(tqdm(dataset, total=len(dataset), dynamic_ncols=True)):
            original = x[self.ori_idx]
            reduced = x[self.reduced_idx]

            ori_ids, ori_seg, ori_mask = self.bert_tokenizer(original, max_length=self.max_len, padding='max_length', return_tensors='np').values()
            ori_length = ori_mask.sum()

            red_ids, red_seg, red_mask = self.bert_tokenizer(reduced, max_length=self.max_len, padding='max_length', return_tensors='np').values()
            red_length = red_mask.sum()

            # Except CLS, SEP (at last)
            original_valid_ids = ori_ids[0, 1:ori_length-1]
            reduced_valid_ids = red_ids[0, 1:red_length-1]

            # valid label: 겹치는 것 1 - 안겹치는 것
            label = np.zeros((self.max_len), np.float)
            _label = np.isin(original_valid_ids, reduced_valid_ids)
            label[1:ori_length-1] = _label

            mask = np.zeros((self.max_len), dtype=np.float)
            mask[1:ori_length-1] = 1

            processed.append({
                'original': original,
                'original_ids': ori_ids[0],
                'original_seg': ori_seg[0],
                'original_valid_length': ori_length,
                'reduced': reduced,
                'reduced_ids': red_ids[0],
                'reduced_seg': red_seg[0],
                'reduced_valid_length': red_length,
                'label': label,
                'mask': mask,
                'augment_mask': 0,
                'query_idx' : idx
            })
            idx+=1

            if self.debug and len(processed) >= 1000:
                break
        

        return processed

    def reset(self):
        self._data = self._base_data

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return (len(self._data))



##########   pair-based #######################################################################################

class SentencePairLossOnlyDataset(Dataset):
    def __init__(self, datapath, bert_tokenizer, ori_idx, reduced_idx, max_len, num_negatives=1, num_sample=1, is_test=False, debug=False, **args):
        self.datapath = datapath
        self.ori_idx = ori_idx
        self.reduced_idx = reduced_idx
        self.max_len = max_len
        self.bert_tokenizer = bert_tokenizer
        self.num_negatives = num_negatives
        self.num_sample = num_sample
        self.is_test = is_test
        self.neg_per_pos = 1.0
        self.debug = debug



        self._candidate_generators = self.positive_pairs()
        self._data = None


        self.update_negative_pairs()


    def positive_pairs(self):
        processed = []
        candidate_generators = {}
        index = 0 

        def _generator(query_idx, original,reduced, candidates):
            random.shuffle(candidates)
            idx = -1
            while True:
                
                idx = (idx + 1) % len(candidates)
                cand_str, cand_type = candidates[idx]
                yield original, reduced, cand_str, cand_type
        

        def _test_generator(original, reduced, candidates):
            # random.shuffle(candidates)
            _types = {'same_length':False, 'subset':False, 'superset':False}
            test_negs = []
            for cand in candidates:
                cand_str, cand_type = cand
                if not _types[cand_type]:
                    test_negs.append((original,reduced, cand_str, cand_type))
                    _types[cand_type] = True
                if _types['same_length'] and _types['subset'] and _types['superset']:
                    break
            return test_negs


        dataset = nlp.data.TSVDataset(self.datapath, field_indices=[self.ori_idx, self.reduced_idx], num_discard_samples=0)
        for i, x in enumerate(tqdm(dataset, total=len(dataset))):
            original = x[self.ori_idx]
            reduced = x[self.reduced_idx]

            negative_candidates = self.generate_candidates(original, reduced)


            if len(negative_candidates) > 0:
                if self.is_test:
                    candidate_generators[(original, reduced)] = _test_generator(original,reduced, negative_candidates)
                else:
                    candidate_generators[(index, original, reduced)] = _generator(index, original, reduced, negative_candidates)
            
            index += 1
            

            if self.debug and i >= 1000:
                break

        return candidate_generators

  


    def generate_candidates(self, original, core):
        candidates = []

        # candidates with same length
        original_tokens = original.split()
        core_tokens = core.split()
        original_ids = [i for i in range(len(original_tokens))]
        core_ids = [i for i in range(len(original_tokens)) if original_tokens[i] in core_tokens]
        non_core_ids = [i for i in range(len(original_tokens)) if i not in core_ids]
        base_mask = np.zeros(len(original_tokens))
        base_mask[core_ids] = 1

        ## random negative ##
        from itertools import combinations
        if len(original_tokens)<3:
            for comb in combinations(original_tokens, 1):
                subsamples = list(comb)
                if ' '.join(subsamples)==core:
                    continue
                else:
                    for num in range(2):
                        candidates.append((' '.join(subsamples), 'random'))

        else:
            for i in range(1,len(original_tokens)):    
                if len(original_tokens)<10:
                    for comb in combinations(original_tokens, i):
                        subsamples = list(comb)
                        if ' '.join(subsamples)==core:
                            continue
                        else:
                            candidates.append((' '.join(subsamples), 'random'))
                else:
                    for num in range(5):
                        sample_idx = sorted(random.sample(original_ids, i))
                        subsamples = [original_tokens[i] for i in sample_idx]
                        if ' '.join(subsamples)==core:
                            sample_idx = sorted(random.sample(original_ids, i))
                            subsamples = [original_tokens[i] for i in sample_idx]
                            candidates.append((' '.join(subsamples), 'random'))
                        else:
                            candidates.append((' '.join(subsamples), 'random'))


        return candidates

    def update_negative_pairs(self):

        negative_samples = []
        if self.is_test:
            for key in tqdm(self._candidate_generators, total=len(self._candidate_generators), desc='Generating negative samples'):
                for neg in self._candidate_generators[key]:
                    original, reduced, negative, neg_type = neg
                    token_ids, segment_ids, attention_mask = self.bert_tokenizer(original, reduced, max_length=self.max_len, padding='max_length', return_tensors='np').values()
                    valid_length = attention_mask.sum()
                    cand_token_ids, cand_segment_ids, cand_attention_mask = self.bert_tokenizer(original, negative, max_length=self.max_len, padding='max_length', return_tensors='np').values()
                    cand_valid_length = cand_attention_mask.sum()
                    negative_samples.append({
                        'original': original,
                        'reduced': reduced,
                        'negative': negative,
                        'token_ids': token_ids,
                        'valid_length': valid_length,
                        'segment_ids': segment_ids,
                        'cand_token_ids': cand_token_ids,
                        'cand_valid_length': cand_valid_length,
                        'cand_segment_ids': cand_segment_ids,
                        'type': neg_type
                    })
        else:
            for key in tqdm(self._candidate_generators, total=len(self._candidate_generators), desc='Generating negative samples'):
                cand_token_ids_list, cand_valid_length_list, cand_segment_ids_list = [], [], []
                for _ in range(self.num_negatives):
                    original, reduced, negative, neg_type = next(self._candidate_generators[key])
                    if neg_type != False:
                        token_ids, segment_ids, attention_mask = self.bert_tokenizer(original, reduced, max_length=self.max_len, padding='max_length', return_tensors='np').values()
                        valid_length = attention_mask.sum()
                        token_ids, segment_ids = token_ids[0], segment_ids[0]
                        cand_token_ids, cand_segment_ids, cand_attention_mask = self.bert_tokenizer(original, negative, max_length=self.max_len, padding='max_length', return_tensors='np').values()
                        cand_valid_length = cand_attention_mask.sum()
                        cand_token_ids, cand_segment_ids = cand_token_ids[0], cand_segment_ids[0]
                        cand_token_ids_list.append(cand_token_ids)
                        cand_valid_length_list.append(cand_valid_length)
                        cand_segment_ids_list.append(cand_segment_ids)
                
                negative_samples.append({
                    'original': original,
                    'reduced': reduced,
                    'negative': negative,
                    'token_ids': token_ids,
                    'valid_length': valid_length,
                    'segment_ids': segment_ids,
                    'cand_token_ids': cand_token_ids_list,
                    'cand_valid_length': cand_valid_length_list,
                    'cand_segment_ids': cand_segment_ids_list,
                    'type': neg_type
                })
            
            
        self._data = negative_samples


    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data) if self._data else len(self._pos_pairs) * (self.num_negatives + 1)



class SentencePairDataset(Dataset):

    def __init__(self, datapath, bert_tokenizer, ori_idx, reduced_idx, max_len, num_negatives=1, num_sample=1, is_test=False, debug=False, **args):
        self.datapath = datapath
        self.ori_idx = ori_idx
        self.reduced_idx = reduced_idx
        self.max_len = max_len
        self.bert_tokenizer = bert_tokenizer
        self.num_negatives = num_negatives
        self.num_sample = num_sample
        self.is_test = is_test
        self.neg_per_pos = 1.0
        self.debug = debug


        # self._data = self.setup()
        
        self._pos_pairs, self._candidate_generators = self.positive_pairs()
        self._data = None
        

        self.update_negative_pairs()


    def positive_pairs(self):
        processed = []
        candidate_generators = {}
        idx = 0

        def _generator(original, candidates, query_idx):
            random.shuffle(candidates)
            idx = -1
            while True:
                idx = (idx + 1) % len(candidates)
                cand_str, cand_type = candidates[idx]

                query_idx = query_idx
                yield original, cand_str, cand_type, query_idx


        def _test_generator(original, candidates):
            # random.shuffle(candidates)
            _types = {'same_length':False, 'subset':False, 'superset':False}
            test_negs = []
            for cand in candidates:
                cand_str, cand_type = cand
                if not _types[cand_type]:
                    test_negs.append((original, cand_str, cand_type))
                    _types[cand_type] = True
                if _types['same_length'] and _types['subset'] and _types['superset']:
                    break
            return test_negs


        idx =0
        dataset = nlp.data.TSVDataset(self.datapath, field_indices=[self.ori_idx, self.reduced_idx], num_discard_samples=0)
        for i, x in enumerate(tqdm(dataset, total=len(dataset), desc='Generating positive samples')):
            original = x[self.ori_idx]
            reduced = x[self.reduced_idx]

            # positive pair
            token_ids, segment_ids, attention_mask = self.bert_tokenizer(original, reduced, max_length=self.max_len, padding='max_length', return_tensors='np').values()
            valid_length = attention_mask.sum()
            token_ids, segment_ids = token_ids[0], segment_ids[0]
            ori_token_ids, ori_segment_ids, ori_attention_mask = self.bert_tokenizer(original, max_length=self.max_len, padding='max_length', return_tensors='np').values()
            ori_valid_length = ori_attention_mask.sum()
            reduced_token_ids, reduced_segment_ids, reduced_attention_mask = self.bert_tokenizer(reduced, max_length=self.max_len, padding='max_length', return_tensors='np').values()
            reduced_valid_length = reduced_attention_mask.sum()
            
            processed.append({
                'original': original,
                'original_tok': ori_token_ids[0],
                'original_valid_length': ori_valid_length,
                'original_segment_ids': ori_segment_ids[0],
                'reduced': reduced,
                'reduced_tok': reduced_token_ids[0],
                'reduced_valid_length': reduced_valid_length,
                'reduced_segment_ids': reduced_segment_ids[0],
                'token_ids': token_ids,
                'valid_length': valid_length,
                'segment_ids': segment_ids,
                'label': 1.0,
                'type': 'positive',
                'query_idx' : idx
            })

            assert len(token_ids) == self.max_len
            assert len(segment_ids) == self.max_len

            negative_candidates = self.generate_candidates(original, reduced)

            if len(negative_candidates) > 0:
                # if candidate_generators.get((original, reduced)) is not None:
                #     print((original, reduced))
                #     exit()

                if self.is_test:
                    candidate_generators[(original, reduced)] = _test_generator(original, negative_candidates)
                else:
                    # index 겹칠 수 있음.
                    candidate_generators[(idx, original, reduced)] = _generator(original, negative_candidates,idx)
            
            idx+=1
            

            if self.debug and len(processed) >= 10000:
                break

        return processed, candidate_generators



    def generate_candidates(self, original, core):
        candidates = []

        # candidates with same length
        original_tokens = original.split()
        core_tokens = core.split()
        core_ids = [i for i in range(len(original_tokens)) if original_tokens[i] in core_tokens]
        non_core_ids = [i for i in range(len(original_tokens)) if i not in core_ids]
        base_mask = np.zeros(len(original_tokens))
        base_mask[core_ids] = 1

        for core, non_core in itertools.product(core_ids, non_core_ids):
            mask = base_mask.copy()
            mask[core] = 0
            mask[non_core] = 1
            candidate_idx = mask.nonzero()[0]
            candidate_tokens = [original_tokens[i] for i in candidate_idx]
            candidates.append((' '.join(candidate_tokens), 'same_length'))

        # candidates with super-set
        if len(non_core_ids) > 1:
            for nc_idx in non_core_ids:
                mask = base_mask.copy()
                mask[nc_idx] = 1
                candidate_idx = mask.nonzero()[0]
                candidate_tokens = [original_tokens[i] for i in candidate_idx]
                candidates.append((' '.join(candidate_tokens), 'superset'))

        # candidates with sub-set
        if len(core_tokens) > 1:
            for core in core_tokens:
                subsamples = [w for w in core_tokens if w != core]
                candidates.append((' '.join(subsamples), 'subset'))

        return candidates

    def update_negative_pairs(self):
        negative_samples = []
        if self.is_test:
            for key in tqdm(self._candidate_generators, total=len(self._candidate_generators), desc='Generating negative samples'):
                for neg in self._candidate_generators[key]:
                    original, negative, neg_type = neg

                    token_ids, segment_ids, attention_mask = self.bert_tokenizer(original, negative, max_length=self.max_len, padding='max_length', return_tensors='np').values()
                    valid_length = attention_mask.sum()
                    token_ids, segment_ids = token_ids[0], segment_ids[0]
                    ori_token_ids, ori_segment_ids, ori_attention_mask = self.bert_tokenizer(original, max_length=self.max_len, padding='max_length', return_tensors='np').values()
                    ori_valid_length = ori_attention_mask.sum()
                    reduced_token_ids, reduced_segment_ids, reduced_attention_mask = self.bert_tokenizer(negative, max_length=self.max_len, padding='max_length', return_tensors='np').values()
                    reduced_valid_length = reduced_attention_mask.sum()
                    
                    negative_samples.append({
                        'original': original,
                        'original_tok': ori_token_ids[0],
                        'original_valid_length': ori_valid_length,
                        'original_segment_ids': ori_segment_ids[0],
                        'reduced': negative,
                        'reduced_tok': reduced_token_ids[0],
                        'reduced_valid_length': reduced_valid_length,
                        'reduced_segment_ids': reduced_segment_ids[0],
                        'token_ids': token_ids,
                        'valid_length': valid_length,
                        'segment_ids': segment_ids,
                        'label': 0.0,
                        'type': neg_type,
                        'query_idx': -1
                    })
        else:
            print('negative')
            for key in tqdm(self._candidate_generators, total=len(self._candidate_generators), desc='Generating negative samples'):
                for _ in range(self.num_negatives):
                    original, negative, neg_type, neg_query_idx = next(self._candidate_generators[key])

                    token_ids, segment_ids, attention_mask = self.bert_tokenizer(original, negative, max_length=self.max_len, padding='max_length', return_tensors='np').values()
                    valid_length = attention_mask.sum()
                    token_ids, segment_ids = token_ids[0], segment_ids[0]
                    ori_token_ids, ori_segment_ids, ori_attention_mask = self.bert_tokenizer(original, max_length=self.max_len, padding='max_length', return_tensors='np').values()
                    ori_valid_length = ori_attention_mask.sum()
                    reduced_token_ids, reduced_segment_ids, reduced_attention_mask = self.bert_tokenizer(negative, max_length=self.max_len, padding='max_length', return_tensors='np').values()
                    reduced_valid_length = reduced_attention_mask.sum()
                        
                    negative_samples.append({
                        'original': original,
                        'original_tok': ori_token_ids[0],
                        'original_segment_ids': ori_segment_ids[0],
                        'original_valid_length': ori_valid_length,
                        'reduced': negative,
                        'reduced_tok': reduced_token_ids[0],
                        'reduced_segment_ids': reduced_segment_ids[0],
                        'reduced_valid_length': reduced_valid_length,
                        'token_ids': token_ids,
                        'valid_length': valid_length,
                        'segment_ids': segment_ids,
                        'label': 0.0,
                        'type': neg_type,
                        'query_idx':neg_query_idx
                    })
        self._data = self._pos_pairs + negative_samples
        self.neg_per_pos = len(negative_samples) / len(self._pos_pairs)

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data) if self._data else len(self._pos_pairs) * (self.num_negatives + 1)
    
    
    
    
class BiencoderPairLossDataset(Dataset):
    def __init__(self, datapath, bert_tokenizer, ori_idx, reduced_idx, max_len, num_negatives=1, num_sample=1, is_test=False, debug=False, **args):
        self.datapath = datapath
        self.ori_idx = ori_idx
        self.reduced_idx = reduced_idx
        self.max_len = max_len
        self.bert_tokenizer = bert_tokenizer
        self.num_negatives = num_negatives
        self.num_sample = num_sample
        self.is_test = is_test
        self.neg_per_pos = 1.0
        self.debug = debug


        self._candidate_generators = self.positive_pairs()
        self._data = None


        self.update_negative_pairs()


    def positive_pairs(self):
        processed = []
        candidate_generators = {}
        index = 0 

        def _generator(query_idx, original,reduced, candidates):
            random.shuffle(candidates)
            idx = -1
            while True:
                
                idx = (idx + 1) % len(candidates)
                cand_str, cand_type = candidates[idx]
                yield original, reduced, cand_str, cand_type
        

        def _test_generator(original, reduced, candidates):
            # random.shuffle(candidates)
            _types = {'same_length':False, 'subset':False, 'superset':False}
            test_negs = []
            for cand in candidates:
                cand_str, cand_type = cand
                if not _types[cand_type]:
                    test_negs.append((original,reduced, cand_str, cand_type))
                    _types[cand_type] = True
                if _types['same_length'] and _types['subset'] and _types['superset']:
                    break
            return test_negs


        dataset = nlp.data.TSVDataset(self.datapath, field_indices=[self.ori_idx, self.reduced_idx], num_discard_samples=0)
        for i, x in enumerate(tqdm(dataset, total=len(dataset), desc='Generating positive samples')):
            original = x[self.ori_idx]
            reduced = x[self.reduced_idx]

            negative_candidates = self.generate_candidates(original, reduced)


            if len(negative_candidates) > 0:
                if self.is_test:
                    candidate_generators[(original, reduced)] = _test_generator(original,reduced, negative_candidates)
                else:
                    candidate_generators[(index, original, reduced)] = _generator(index, original, reduced, negative_candidates)
            
            index += 1
            

            if self.debug and i >= 1000:
                break

        return candidate_generators

  


    def generate_candidates(self, original, core):
        candidates = []

        # candidates with same length
        original_tokens = original.split()
        core_tokens = core.split()
        core_ids = [i for i in range(len(original_tokens)) if original_tokens[i] in core_tokens]
        non_core_ids = [i for i in range(len(original_tokens)) if i not in core_ids]
        base_mask = np.zeros(len(original_tokens))
        base_mask[core_ids] = 1

        for core, non_core in itertools.product(core_ids, non_core_ids):
            mask = base_mask.copy()
            mask[core] = 0
            mask[non_core] = 1
            candidate_idx = mask.nonzero()[0]
            candidate_tokens = [original_tokens[i] for i in candidate_idx]
            candidates.append((' '.join(candidate_tokens), 'same_length'))

        # candidates with super-set
        if len(non_core_ids) > 1:
            for nc_idx in non_core_ids:
                mask = base_mask.copy()
                mask[nc_idx] = 1
                candidate_idx = mask.nonzero()[0]
                candidate_tokens = [original_tokens[i] for i in candidate_idx]
                candidates.append((' '.join(candidate_tokens), 'superset'))

        # candidates with sub-set
        if len(core_tokens) > 1:
            for core in core_tokens:
                subsamples = [w for w in core_tokens if w != core]
                candidates.append((' '.join(subsamples), 'subset'))

        return candidates

    def update_negative_pairs(self):

        negative_samples = []
        if self.is_test:
            for key in tqdm(self._candidate_generators, total=len(self._candidate_generators), desc='Generating negative samples'):
                for neg in self._candidate_generators[key]:
                    original, reduced, negative, neg_type = neg

                    ori_token_ids, ori_segment_ids, ori_attention_mask = self.bert_tokenizer(original, max_length=self.max_len, padding='max_length', return_tensors='np').values()
                    pos_token_ids, pos_segment_ids, pos_attention_mask = self.bert_tokenizer(reduced, max_length=self.max_len, padding='max_length', return_tensors='np').values()
                    neg_token_ids, neg_segment_ids, neg_attention_mask = self.bert_tokenizer(negative, max_length=self.max_len, padding='max_length', return_tensors='np').values()
                    ori_valid_length = ori_attention_mask.sum()
                    pos_valid_length = pos_attention_mask.sum()
                    neg_valid_length = neg_attention_mask.sum()

                    negative_samples.append({
                        'original': original,
                        'original_tok': ori_token_ids[0],
                        'original_segment_ids': ori_segment_ids[0],
                        'original_valid_length': ori_valid_length,
                        'reduced': reduced,
                        'pos_tok': pos_token_ids[0],
                        'pos_segment_ids': pos_segment_ids[0],
                        'pos_valid_length': pos_valid_length,
                        'negative': negative,
                        'neg_tok': neg_token_ids[0],
                        'neg_segment_ids': neg_segment_ids[0],
                        'neg_valid_length': neg_valid_length,
                        'token_ids': ori_token_ids[0],
                        'valid_length': ori_valid_length,
                        'segment_ids': ori_segment_ids[0],
                        'type': neg_type
                    })
        else:
            for key in tqdm(self._candidate_generators, total=len(self._candidate_generators), desc='Generating negative samples'):
                neg_token_ids_list, neg_segment_ids_list, neg_valid_length_list = [], [], []
                for _ in range(self.num_negatives):
                    original, reduced, negative, neg_type = next(self._candidate_generators[key])
                    if neg_type != False:    

                        ori_token_ids, ori_segment_ids, ori_attention_mask = self.bert_tokenizer(original, max_length=self.max_len, padding='max_length', return_tensors='np').values()
                        ori_valid_length = ori_attention_mask.sum()
                        pos_token_ids, pos_segment_ids, pos_attention_mask = self.bert_tokenizer(reduced, max_length=self.max_len, padding='max_length', return_tensors='np').values()
                        pos_valid_length = pos_attention_mask.sum()
                        neg_token_ids, neg_segment_ids, neg_attention_mask = self.bert_tokenizer(negative, max_length=self.max_len, padding='max_length', return_tensors='np').values()
                        neg_valid_length = neg_attention_mask.sum()
                        neg_token_ids_list.append(neg_token_ids[0])
                        neg_segment_ids_list.append(neg_segment_ids[0])
                        neg_valid_length_list.append(neg_valid_length)
          
                negative_samples.append({
                    'original': original,
                    'original_tok': ori_token_ids[0],
                    'original_segment_ids': ori_segment_ids[0],
                    'original_valid_length': ori_valid_length,
                    'reduced': reduced,
                    'pos_tok': pos_token_ids[0],
                    'pos_segment_ids': pos_segment_ids[0],
                    'pos_valid_length': pos_valid_length,
                    'negative': negative,
                    'neg_tok': neg_token_ids_list,
                    'neg_segment_ids': neg_segment_ids_list,
                    'neg_valid_length': neg_valid_length_list,
                    'token_ids': ori_token_ids[0],
                    'valid_length': ori_valid_length,
                    'segment_ids': ori_segment_ids[0],
                    'type': neg_type
                })
            
            
        self._data = negative_samples


    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data) if self._data else len(self._pos_pairs) * (self.num_negatives + 1)    



class GenerateDataset(Dataset):
    def __init__(self, datapath, bert_tokenizer, ori_idx, reduced_idx, max_len, num_negatives=1, num_sample=1, is_test=False, debug=False, **args):
        self.datapath = datapath
        self.ori_idx = ori_idx
        self.reduced_idx = reduced_idx
        self.max_len = max_len
        self.bert_tokenizer = bert_tokenizer
        
        self._pos_pairs, self._candidate_generators = self.positive_pairs()
        self._data = self._pos_pairs

    def positive_pairs(self):
        processed = []
        candidate_generators = {}
        idx = 0
        dataset = nlp.data.TSVDataset(self.datapath, field_indices=[self.ori_idx, self.reduced_idx], num_discard_samples=0)
        for i, x in enumerate(tqdm(dataset, total=len(dataset), desc='Loading dataset')):
            original = x[self.ori_idx]
            reduced = x[self.reduced_idx]

            processed.append({
                'original': original,
                'reduced': reduced,
                'label': 1.0,
                'type': 'positive',
                'query_idx' : idx
            })

            idx+=1

        return processed, candidate_generators

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data) if self._data else len(self._pos_pairs)