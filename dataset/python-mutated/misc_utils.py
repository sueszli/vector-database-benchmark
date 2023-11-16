import copy
import functools
import os
import random
import subprocess
import weakref
from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING
import numpy
import torch
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import PROC_COLUMN
from ludwig.globals import DESCRIPTION_FILE_NAME
from ludwig.utils import fs_utils
from ludwig.utils.fs_utils import find_non_existing_dir_by_adding_suffix
if TYPE_CHECKING:
    from ludwig.schema.model_types.base import ModelConfig

@DeveloperAPI
def set_random_seed(random_seed):
    if False:
        for i in range(10):
            print('nop')
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    numpy.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

@DeveloperAPI
def merge_dict(dct, merge_dct):
    if False:
        i = 10
        return i + 15
    'Recursive dict merge. Inspired by :meth:``dict.update()``, instead of updating only top-level keys,\n    dict_merge recurses down into dicts nested to an arbitrary depth, updating keys. The ``merge_dct`` is merged\n    into ``dct``.\n\n    :param dct: dict onto which the merge is executed\n    :param merge_dct: dct merged into dct\n    :return: None\n    '
    dct = copy.deepcopy(dct)
    for (k, v) in merge_dct.items():
        if k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], Mapping):
            dct[k] = merge_dict(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct

@DeveloperAPI
def sum_dicts(dicts, dict_type=dict):
    if False:
        return 10
    summed_dict = dict_type()
    for d in dicts:
        for (key, value) in d.items():
            if key in summed_dict:
                prev_value = summed_dict[key]
                if isinstance(value, (dict, OrderedDict)):
                    summed_dict[key] = sum_dicts([prev_value, value], dict_type=type(value))
                elif isinstance(value, numpy.ndarray):
                    summed_dict[key] = numpy.concatenate((prev_value, value))
                else:
                    summed_dict[key] = prev_value + value
            else:
                summed_dict[key] = value
    return summed_dict

@DeveloperAPI
def get_from_registry(key, registry):
    if False:
        print('Hello World!')
    if hasattr(key, 'lower'):
        key = key.lower()
    if key in registry:
        return registry[key]
    else:
        raise ValueError(f"Key '{key}' not in registry, available options: {registry.keys()}")

@DeveloperAPI
def set_default_value(dictionary, key, value):
    if False:
        i = 10
        return i + 15
    if key not in dictionary:
        dictionary[key] = value

@DeveloperAPI
def set_default_values(dictionary: dict, default_value_dictionary: dict):
    if False:
        i = 10
        return i + 15
    'This function sets multiple default values recursively for various areas of the config. By using the helper\n    function set_default_value, It parses input values that contain nested dictionaries, only setting values for\n    parameters that have not already been defined by the user.\n\n    Args:\n        dictionary (dict): The dictionary to set default values for, generally a section of the config.\n        default_value_dictionary (dict): The dictionary containing the default values for the config.\n    '
    for (key, value) in default_value_dictionary.items():
        if key not in dictionary:
            dictionary[key] = value
        elif value == {}:
            set_default_value(dictionary, key, value)
        elif isinstance(value, dict) and value:
            set_default_values(dictionary[key], value)
        else:
            set_default_value(dictionary, key, value)

@DeveloperAPI
def get_class_attributes(c):
    if False:
        print('Hello World!')
    return {i for i in dir(c) if not callable(getattr(c, i)) and (not i.startswith('_'))}

@DeveloperAPI
def get_output_directory(output_directory, experiment_name, model_name='run'):
    if False:
        print('Hello World!')
    base_dir_name = os.path.join(output_directory, experiment_name + ('_' if model_name else '') + (model_name or ''))
    return fs_utils.abspath(find_non_existing_dir_by_adding_suffix(base_dir_name))

@DeveloperAPI
def get_file_names(output_directory):
    if False:
        while True:
            i = 10
    description_fn = os.path.join(output_directory, DESCRIPTION_FILE_NAME)
    training_stats_fn = os.path.join(output_directory, 'training_statistics.json')
    model_dir = os.path.join(output_directory, 'model')
    return (description_fn, training_stats_fn, model_dir)

@DeveloperAPI
def get_combined_features(config):
    if False:
        print('Hello World!')
    return config['input_features'] + config['output_features']

@DeveloperAPI
def get_proc_features(config):
    if False:
        while True:
            i = 10
    return get_proc_features_from_lists(config['input_features'], config['output_features'])

@DeveloperAPI
def get_proc_features_from_lists(*args):
    if False:
        for i in range(10):
            print('nop')
    return {feature[PROC_COLUMN]: feature for features in args for feature in features}

@DeveloperAPI
def set_saved_weights_in_checkpoint_flag(config_obj: 'ModelConfig'):
    if False:
        print('Hello World!')
    'Adds a flag to all input feature encoder configs indicating that the weights are saved in the checkpoint.\n\n    Next time the model is loaded we will restore pre-trained encoder weights from ludwig model (and not load from cache\n    or model hub).\n    '
    for input_feature in config_obj.input_features:
        encoder_obj = input_feature.encoder
        encoder_obj.saved_weights_in_checkpoint = True

@DeveloperAPI
def remove_empty_lines(str):
    if False:
        print('Hello World!')
    return '\n'.join([line.rstrip() for line in str.split('\n') if line.rstrip()])

@DeveloperAPI
def memoized_method(*lru_args, **lru_kwargs):
    if False:
        print('Hello World!')

    def decorator(func):
        if False:
            print('Hello World!')

        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            if False:
                return 10
            self_weak = weakref.ref(self)

            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                return func(self_weak(), *args, **kwargs)
            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)
        return wrapped_func
    return decorator

@DeveloperAPI
def get_commit_hash():
    if False:
        i = 10
        return i + 15
    'If Ludwig is run from a git repository, get the commit hash of the current HEAD.\n\n    Returns None if git is not executable in the current environment or Ludwig is not run in a git repo.\n    '
    try:
        with open(os.devnull, 'w') as devnull:
            is_a_git_repo = subprocess.call(['git', 'branch'], stderr=subprocess.STDOUT, stdout=devnull) == 0
        if is_a_git_repo:
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8')
            return commit_hash
    except:
        pass
    return None