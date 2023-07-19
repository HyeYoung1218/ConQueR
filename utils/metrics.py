import numpy as np
from collections import OrderedDict
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix

def accuracy(y_pred, y_true, mask=None, reduce=True):
    return accuracy_score(y_true, y_pred)

def exact_match(y_pred, y_true, mask, reduce=True):
    token_correct = (y_pred == y_true) * mask
    em_samples = token_correct.sum(1) == mask.sum(1)
    if reduce:
        return em_samples.sum()
    else:
        return em_samples

def batch_masked_cm(y_pred, y_true, mask, reduce=True):
    y_pred *= mask
    y_true *= mask
    _batch_cm = multilabel_confusion_matrix(y_true, y_pred, samplewise=True).sum(0)
    num_mask = (1 - mask).sum()
    _batch_cm[0, 0] -= num_mask
    if reduce:
        _batch_cm = multilabel_confusion_matrix(y_true, y_pred, samplewise=True).sum(0)
        num_mask = (1 - mask).sum()
        _batch_cm[0, 0] -= num_mask
    else:
        _batch_cm = multilabel_confusion_matrix(y_true, y_pred, samplewise=True)
        batch_num_mask = (1 - mask).sum(1)
        _batch_cm[:, 0, 0] -= batch_num_mask.astype(int)

    return _batch_cm

def precision_recall(confusion_matrix):
    prec = confusion_matrix / confusion_matrix.sum(0, keepdims=True)
    neg_prec, pos_prec = prec[0, 0], prec[1, 1]
    recall = confusion_matrix / confusion_matrix.sum(1, keepdims=True)
    neg_recall, pos_recall = recall[0, 0], recall[1, 1]

    return OrderedDict({
        'neg_prec': neg_prec if not np.isnan(neg_prec) else 0.0,
        'neg_recall': neg_recall if not np.isnan(neg_recall) else 0.0,
        'pos_prec': pos_prec if not np.isnan(pos_prec) else 0.0,
        'pos_recall': pos_recall if not np.isnan(pos_recall) else 0.0
    })


__all__ = [
    'exact_match',
    'mcm',
    'precision_recall'
]

if __name__ == '__main__':
    # T [0, 1, 1, 0, 1]
    # P [0, 1, 1, 1, 1]
    
    ##### manual
    # neg_precision = (# neg correct / # neg pred)  = 1 / 1
    # neg_recall = (# neg correct / # neg true)     = 1 / 2
    # pos_precision = (# pos correct / # pos pred)  = 3 / 4
    # pos_recall = (# pos correct / # pos true)     = 3 / 3

    y_true = np.array([
        [0, 1, 1, 0, 1],
        [0, 1, 1, 1, 1],
        [1, 1, 0, 1, 1]
    ])
    y_pred = np.array([
        [1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ])
    mask = np.array([
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0]
    ])
    # mask = np.array([
    #     [0, 1, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 1, 1, 1, 0, 0, 0, 0, 0],
    #     [0, 1, 1, 1, 0, 0, 0, 0, 0]
    # ])
    # y_true = np.array([
    #     [0, 1, 1, 0, 1, 0, 0, 0, 0],
    #     [0, 1, 1, 1, 1, 0, 0, 0, 0],
    #     [1, 1, 0, 1, 1, 0, 0, 0, 0]
    # ])
    # y_pred = np.array([
    #     [1, 1, 1, 0, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1]
    # ])
    # print(exact_match(y_pred, y_true, mask) / len(y_true))

    # correct = (y_pred == y_true) * mask
    # query_correct = correct.sum(1) == mask.sum(1)
    # print(query_correct.sum() / len(y_true))
    # print(precision_recall(mcm(y_pred, y_true).sum(0)))
    print(batch_masked_cm(y_pred, y_true, mask))
