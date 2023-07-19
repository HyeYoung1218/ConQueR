import os
from typing import List
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from collections import OrderedDict
from IPython import embed


@dataclass
class BaseConfig:
    save_model: bool = True
    save_dir: str = 'saves'
    tensorboard: bool = False
    gpus: str = '0'
    # name of the experiment: used as direectory names under 'save_dir'
    exp_name: str = 'sample'
    verbose:int = 0
    seed:int = 2021
    # model save interval
    ckpt_interval:int = 1
    # validation interval
    eval_interval:int = 1

@dataclass
class ExpConfig:
    batch_size: int = 32
    num_epochs: int = 5
    learning_rate: float = 1e-5
    threshold: float = 0.5
    warmup_ratio: float = 0.2
    max_grad_norm: int = 1
    truncated_loss: bool = True
    forget_rate: float = 0.3
    num_gradual: int = 3
    exponent: float = 2
    prune_epoch: int = 1
    probability: float = 0.4
    n_components: int = 2
    max_iter: int = 100
    tol: float = 1e-3
    reg_covar: float = 1e-6

@dataclass
class DatasetConfig:

    datadir: str = './data'
    debug: bool = False
    max_len:int = 120
    phr_sep: bool = False
    phr_sep_token: str = '[SEP]'
    augment_type: str = 'recon'
    augment_ratio: float = 0.0
    augment_lambda: float = 1.0
    num_truncate: int = 1
    num_negatives: int = 1
    num_sample: int = 1

@dataclass
class ModelConfig:
    bert_model: str = 'monologg/koelectra-base-v3-discriminator' ## monologg/koelectra-base-v3-discriminator, etribert
    fc_layers: list = field(default_factory=lambda: [])
    dropout: float = 0.2
    cls_enhanced: bool = False
    pred_num_core: bool = False
    core_lambda: float = 0.0
    contrastive: bool = False
    cont_temp: float = 0.01
    cont_lambda: float = 1.0
    tree_transformer: bool = False
    num_tree_layers: int = 1
    num_tree_heads: int = 12
    truncated_loss: bool = True
    pairwise: bool = True
    

@dataclass
class PairDatasetConfig:
    # 위와 동일
    datadir: str = './data'
    dataset: str = 'naver'
    debug: bool = False
    # pair라서 2배 해줌
    max_len:int = 120
    num_negatives: int = 1
    num_sample: int = 1

@dataclass
class PairModelConfig:
    bert_model: str = 'monologg/koelectra-base-v3-discriminator' ## monologg/koelectra-base-v3-discriminator, etribert
    fc_layers: list = field(default_factory=lambda: [])
    dropout: float = 0.2
    pooling: bool = False
    pairwise: bool = False
    

def load_config():
    base_conf = OmegaConf.structured({'base' : BaseConfig})
    dataset_conf = OmegaConf.structured({'dataset' : DatasetConfig})
    model_conf = OmegaConf.structured({'model' : ModelConfig})
    exp_conf = OmegaConf.structured({'exp' : ExpConfig})
    # cli_conf = OmegaConf.from_cli()

    conf = OmegaConf.merge(base_conf, dataset_conf, model_conf, exp_conf)

    return conf

def load_pair_config():
    base_conf = OmegaConf.structured({'base' : BaseConfig})
    dataset_conf = OmegaConf.structured({'dataset' : PairDatasetConfig})
    model_conf = OmegaConf.structured({'model' : PairModelConfig})
    exp_conf = OmegaConf.structured({'exp' : ExpConfig})

    conf = OmegaConf.merge(base_conf, dataset_conf, model_conf, exp_conf)

    return conf

def load_config_from_path(yaml_config_path):
    if not os.path.exists(yaml_config_path):
        raise FileNotFoundError(f'Config file not found in {yaml_config_path}')

    with open(yaml_config_path, "r") as f:
        loaded = OmegaConf.load(f)

    return loaded

def ensure_value_type(v):
    BOOLEAN = {'false': False, 'False': False,
                'true': True, 'True': True}
    if isinstance(v, str):
        try:
            value = eval(v)
            if not isinstance(value, (str, int, float, list, tuple)):
                value = v
        except:
            if v in BOOLEAN:
                v = BOOLEAN[v]
            value = v
    else:
        value = v
    return value

def update_params(conf, params):
    # for now, assume 'params' is dictionary
    new_params = OrderedDict()
    new_params.update(params)
    params = new_params
    for k, v in params.items():
        updated=False

        for section in conf.keys():
            if k in conf[section]:
                conf[section][k] = ensure_value_type(v)
                updated = True
                break
    
        if not updated:
            # raise ValueError
            print('Parameter not updated. \'%s\' not exists.' % k)
    return conf


if __name__ == '__main__':
    import pprint
    conf = load_config()

    pprint.pprint(conf)