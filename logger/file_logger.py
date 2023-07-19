import io
import os
import csv
import json
import yaml
from time import strftime
from collections import OrderedDict
from omegaconf import OmegaConf
from argparse import Namespace

from logger.base import Logger

class FileLogger(Logger):
    
    HPARAMS_FILE = 'hparams.yaml'
    LOG_FILE = 'logs.csv'

    def __init__(self, base_dir, exp_name=None):
        self._base_dir = base_dir
        self.exp_name = exp_name
        self._log_dir = None
        self.hparams = {}
        self.metrics_history = None
        self.headers = None
    
    @property
    def base_dir(self):
        if self.exp_name is None:
            directory = self._base_dir
        else:
            directory = os.path.join(self._base_dir, self.exp_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory
    
    @property
    def log_dir(self):
        if self._log_dir is None:
            self._log_dir = self.get_next_log_dir()
        return self._log_dir

    def get_next_log_dir(self):
        existing_exps = os.listdir(self.base_dir)
        used_idx = []
        for exp in existing_exps:
            if os.path.isdir(os.path.join(self.base_dir, exp)) and exp.startswith('exp'):
                idx = int(exp.split('_')[0][3:])
                used_idx.append(idx)
        if len(used_idx) == 0:
            next_idx = 0
        else:
            next_idx = max(used_idx) + 1
        next_log_dir = 'exp%d_%s' % (next_idx, strftime('%Y%m%d-%H%M'))
        next_log_dir = os.path.join(self.base_dir, next_log_dir)
        if not os.path.exists(next_log_dir):
            os.makedirs(next_log_dir)
        return next_log_dir

    def log_hparams(self, hparams):
        self.hparams.update(hparams)
    
    def log_metrics(self, metrics, epoch=None, prefix=None):
        if prefix:
            metrics = self.add_dict_prefix(metrics)
            
        if self.metrics_history is None:
            self.metrics_history = []
            self.headers = list(metrics.keys())
            if 'epoch' not in self.headers:
                self.headers = ['epoch'] + self.headers

        for key in self.headers:
            if key not in metrics:
                metrics[key] = ' - '
        
        for key in metrics:
            if key not in self.headers:
                self.headers.append(key)
        
        if epoch is None:
            epoch = len(self.metrics_history)

        metrics['epoch'] = epoch

        self.metrics_history.append(self.ensure_ordered_dict(metrics))
    
    def check_columns(self, d):
        pass
    
    def ensure_ordered_dict(self, d):
        if isinstance(d, OrderedDict):
            return d
        else:
            return OrderedDict(d)
    
    def save_hparams(self):
        # convert Namespace or AD to dict
        if isinstance(self.hparams, Namespace):
            hparams = vars(self.hparams)
        else:
            hparams = self.hparams
        hparams_file = os.path.join(self.log_dir, self.HPARAMS_FILE)

        # saving with OmegaConf objects
        if OmegaConf.is_config(hparams):
            with open(hparams_file, "w", encoding="utf-8") as fp:
                OmegaConf.save(hparams, fp, resolve=True)
            return
        for v in hparams.values():
            if OmegaConf.is_config(v):
                with open(hparams_file, "w", encoding="utf-8") as fp:
                    OmegaConf.save(OmegaConf.create(hparams), fp, resolve=True)
                return

        assert isinstance(hparams, dict)
        hparams_allowed = {}
        # drop paramaters which contain some strange datatypes as fsspec
        for k, v in hparams.items():
            try:
                yaml.dump(v)
            except TypeError as err:
                print(f"Skipping '{k}' parameter because it is not possible to safely dump to YAML.")
                hparams[k] = type(v).__name__
            else:
                hparams_allowed[k] = v

        # saving the standard way
        with open(hparams_file, "w", newline="") as fp:
            yaml.dump(hparams_allowed, fp)
    
    def save(self):
        """Save recorded hparams and metrics into files"""
        if not self.metrics_history:
            return

        if 'train_loss' in self.headers:
            idx = self.headers.index('train_loss')
            self.headers = [self.headers.pop(idx)] + self.headers
        if 'elapsed' in self.headers:
            idx = self.headers.index('elapsed')
            self.headers = [self.headers.pop(idx)] + self.headers
        if 'epoch' in self.headers:
            idx = self.headers.index('epoch')
            self.headers = [self.headers.pop(idx)] + self.headers
        
        log_file = os.path.join(self.log_dir, self.LOG_FILE)
        with io.open(log_file, 'w', newline='') as f:
            self.writer = csv.DictWriter(f, fieldnames=self.headers)
            self.writer.writeheader()
            self.writer.writerows(self.metrics_history)