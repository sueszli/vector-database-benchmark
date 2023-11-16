from typing import List, Union
from collections import OrderedDict
import glob
import os
from pathlib import Path
import torch

def _load_weights(path: str) -> dict:
    if False:
        print('Hello World!')
    '\n    Load weights of a model.\n\n    Args:\n        path: Path to model weights\n\n    Returns:\n        Weights\n    '
    weights = torch.load(path, map_location=lambda storage, loc: storage)
    if 'model_state_dict' in weights:
        weights = weights['model_state_dict']
    return weights

def average_weights(state_dicts: List[dict]) -> OrderedDict:
    if False:
        print('Hello World!')
    '\n    Averaging of input weights.\n\n    Args:\n        state_dicts: Weights to average\n\n    Raises:\n        KeyError: If states do not match\n\n    Returns:\n        Averaged weights\n    '
    params_keys = None
    for (i, state_dict) in enumerate(state_dicts):
        model_params_keys = list(state_dict.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError('For checkpoint {}, expected list of params: {}, but found: {}'.format(i, params_keys, model_params_keys))
    average_dict = OrderedDict()
    for k in state_dicts[0].keys():
        average_dict[k] = torch.div(sum((state_dict[k] for state_dict in state_dicts)), len(state_dicts))
    return average_dict

def get_averaged_weights_by_path_mask(path_mask: str, logdir: Union[str, Path]=None) -> OrderedDict:
    if False:
        while True:
            i = 10
    '\n    Averaging of input weights and saving them.\n\n    Args:\n        path_mask: globe-like pattern for models to average\n        logdir: Path to logs directory\n\n    Returns:\n        Averaged weights\n    '
    if logdir is None:
        models_pathes = glob.glob(path_mask)
    else:
        models_pathes = glob.glob(os.path.join(logdir, 'checkpoints', path_mask))
    all_weights = [_load_weights(path) for path in models_pathes]
    averaged_dict = average_weights(all_weights)
    return averaged_dict
__all__ = ['average_weights', 'get_averaged_weights_by_path_mask']