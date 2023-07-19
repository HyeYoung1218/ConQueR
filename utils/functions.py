import os
import torch
import random
import numpy as np
from time import strftime, sleep

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_next_log_dir(log_base, exp_name):
    """
    Generate directory path to log

    :param log_dir:

    :return:
    """
    log_dir = os.path.join(log_base, exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dirs = os.listdir(log_dir)
    if len(log_dirs) == 0:
        idx = 0
    else:
        idx_list = sorted([int(d.split('_')[0]) for d in log_dirs])
        idx = idx_list[-1] + 1

    cur_log_dir = '%d_%s' % (idx, strftime('%Y%m%d-%H%M'))
    full_log_dir = os.path.join(log_dir, cur_log_dir)

    if not os.path.exists(full_log_dir):
        os.makedirs(full_log_dir)
    else:
        full_log_dir = generate_next_log_dir(log_dir)

    return full_log_dir