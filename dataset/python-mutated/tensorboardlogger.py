from ray.tune.utils import flatten_dict
import numpy as np
import os
import math
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from bigdl.dllib.utils.log4Error import *
VALID_SUMMARY_TYPES = (int, float)
VALID_NUMPY_SUMMARY_TYPES = (np.float32, np.float64, np.int32)
VALID_SEQ_SUMMARY_TYPES = (list, tuple)

class TensorboardLogger:

    def __init__(self, logs_dir='', writer=None, name='AutoML'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize a tensorboard logger\n\n        Note that this logger relies on tensorboardx and only provide tensorboard hparams log.\n        An ImportError will be raised for the lack of torch\n\n        :param logs_dir: root directory for the log, default to the current working dir\n        :param writer: shared tensorboard SummaryWriter, default to None.\n        '
        self.logs_dir = logs_dir
        self.name = name
        self._file_writer = None
        if writer:
            self._file_writer = writer
        else:
            self._file_writer = SummaryWriter(log_dir=self.logs_dir)

    def run(self, config, metric):
        if False:
            while True:
                i = 10
        '\n        Write log files(event files)\n\n        The log files is arranged as following:\n        self.logs_dir\n        |--eventfile_all\n        |--Trail_1\n        |  |--eventfile_1\n        |--Trail_2\n        |  |--eventfile_2\n        ...\n        :param config: A dictionary. Keys are trail name,\n            value is a dictionary indicates the trail config\n        :param metric: A dictionary. Keys are trail name,\n            value is a dictionary indicates the trail metric results for each iteration\n\n        Example:\n        Config = {"run1":{"lr":0.001, "hidden_units": 32},\n                  "run2":{"lr":0.01, "hidden_units": 64}}\n        Metric = {"run1":{"acc":0.91, "time": 32.13},\n                  "run2":{"acc":[0.93, 0.95], "time": [61.33, 62.44]}}\n\n        Note that the keys of config and metric should be exactly the same\n        '
        invalidInputError(config.keys() == metric.keys(), 'The keys of config and metric should be exactly the same')
        new_config = {}
        hparam_domain_discrete = {}
        for key in config.keys():
            new_config[key] = {}
            for (k, value) in config[key].items():
                if value is None:
                    pass
                if type(value) in VALID_SUMMARY_TYPES:
                    new_config[key][f'{self.name}/' + k] = value
                if type(value) in VALID_NUMPY_SUMMARY_TYPES and (not np.isnan(value)):
                    new_config[key][f'{self.name}/' + k] = float(value)
                if type(value) in VALID_SEQ_SUMMARY_TYPES:
                    new_config[key][f'{self.name}/' + k] = str(value)
                    if f'{self.name}/' + k in hparam_domain_discrete:
                        hparam_domain_discrete[f'{self.name}/' + k].add(str(value))
                    else:
                        hparam_domain_discrete[f'{self.name}/' + k] = set([str(value)])
        for (k, v) in hparam_domain_discrete.items():
            hparam_domain_discrete[k] = list(v)
        for key in new_config.keys():
            if new_config[key] == {}:
                del new_config[key]
        new_metric = {}
        for key in metric.keys():
            new_metric[key] = {}
            for (k, value) in metric[key].items():
                if not isinstance(value, list):
                    value = [value]
                if value[-1] is None:
                    continue
                if type(value[-1]) in VALID_SUMMARY_TYPES and (not math.isnan(value[-1])):
                    new_metric[key][f'{self.name}/' + k] = value
                if type(value[-1]) in VALID_NUMPY_SUMMARY_TYPES and (not np.isnan(value[-1])):
                    new_metric[key][f'{self.name}/' + k] = list(map(float, value))
        for key in new_metric.keys():
            if new_metric[key] == {}:
                del new_metric[key]
        for key in new_metric.keys():
            self._write_hparams(new_config[key], new_metric[key], name=key.replace('/', '_'), hparam_domain_discrete=hparam_domain_discrete)

    def _write_hparams(self, hparam_dict, metric_dict, name, hparam_domain_discrete):
        if False:
            while True:
                i = 10
        (exp, ssi, sei) = hparams(hparam_dict, metric_dict, hparam_domain_discrete)
        w_hp = SummaryWriter(log_dir=os.path.join(self._file_writer.log_dir, name))
        w_hp.file_writer.add_summary(exp)
        w_hp.file_writer.add_summary(ssi)
        w_hp.file_writer.add_summary(sei)
        for (k, values) in metric_dict.items():
            global_step = 0
            for v in values:
                w_hp.add_scalar(k, v, global_step)
                global_step += 1
        w_hp.close()

    @staticmethod
    def _ray_tune_searcher_log_adapt(analysis):
        if False:
            while True:
                i = 10
        config = analysis.get_all_configs()
        metric_raw = analysis.fetch_trial_dataframes()
        metric = {}
        for (key, value) in metric_raw.items():
            metric[key] = dict(zip(list(value.columns), list(map(list, value.values.T))))
            config[key]['address'] = key
        return (config, metric)

    def close(self):
        if False:
            print('Hello World!')
        '\n        Close the logger\n        '
        self._file_writer.close()