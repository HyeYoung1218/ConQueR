from models.BERTReduction import BERTReduction
from models.SentencePairModel import SentencePairModel

def model_builder(model_name='bert', bert=None, device='cpu', **model_args):
    if model_name == 'bert':
        model = BERTReduction(backbone=bert, device=device, **model_args)
    elif model_name == 'bert_pair':
        model = SentencePairModel(backbone=bert, device=device, **model_args)
    else:
        ValueError(f'Incorrect model name specified: {model_name}')
    return model