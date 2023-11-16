import copy
import logging
import math
from collections import OrderedDict
from typing import List
import psutil
try:
    import GPUtil
except ImportError:
    raise ImportError(' ray is not installed. In order to use auto_train please run pip install ludwig[ray]')
from ludwig.api import LudwigModel
from ludwig.backend import initialize_backend
from ludwig.constants import AUTO, AUTOML_DEFAULT_TEXT_ENCODER, AUTOML_LARGE_TEXT_DATASET, AUTOML_MAX_ROWS_PER_CHECKPOINT, AUTOML_SMALLER_TEXT_ENCODER, AUTOML_SMALLER_TEXT_LENGTH, AUTOML_TEXT_ENCODER_MAX_TOKEN_LEN, HYPEROPT, MINIMUM_BATCH_SIZE, PREPROCESSING, SPACE, TEXT, TRAINER
from ludwig.data.preprocessing import preprocess_for_training
from ludwig.features.feature_registries import update_config_with_metadata
from ludwig.schema.model_config import ModelConfig
from ludwig.utils.automl.utils import get_model_type
from ludwig.utils.torch_utils import initialize_pytorch
logger = logging.getLogger(__name__)
RANKED_MODIFIABLE_PARAM_LIST = {'tabnet': OrderedDict({'trainer.batch_size': 32, 'combiner.size': 8, 'combiner.output_size': 8}), 'concat': OrderedDict({'trainer.batch_size': 32, 'combiner.output_size': 64, 'combiner.num_fc_layers': 1}), 'tabtransformer': OrderedDict({'trainer.batch_size': 32, 'combiner.num_heads:': 4, 'combiner.output_size': 8, 'combiner.num_layers': 4, 'combiner.num_fc_layers': 1}), 'text': OrderedDict({'trainer.batch_size': 16})}
BYTES_PER_MiB = 1048576
BYTES_PER_WEIGHT = 4
BYTES_OPTIMIZER_PER_WEIGHT = 8

def get_trainingset_metadata(config, dataset, backend):
    if False:
        for i in range(10):
            print('nop')
    (_, _, _, training_set_metadata) = preprocess_for_training(config, dataset=dataset, preprocessing_params=config[PREPROCESSING], backend=backend)
    return training_set_metadata

def _get_machine_memory():
    if False:
        for i in range(10):
            print('nop')
    if GPUtil.getGPUs():
        machine_mem = GPUtil.getGPUs()[0].memoryTotal * BYTES_PER_MiB
    else:
        machine_mem = psutil.virtual_memory().total
    return machine_mem

def _get_text_feature_max_length(config, training_set_metadata) -> int:
    if False:
        i = 10
        return i + 15
    'Returns max sequence length over text features, subject to preprocessing limit.'
    max_length = 0
    for feature in config['input_features']:
        if feature['type'] == TEXT:
            feature_max_len = training_set_metadata[feature['name']]['max_sequence_length']
            if feature_max_len > max_length:
                max_length = feature_max_len
    if 'preprocessing' in config and TEXT in config['preprocessing'] and ('max_sequence_length' in config['preprocessing'][TEXT]):
        limit = config['preprocessing'][TEXT]['max_sequence_length']
    else:
        limit = 256
    if max_length > limit + 2:
        max_length = limit + 2
    return max_length

def _get_text_model_memory_usage(config, training_set_metadata, memory_usage) -> int:
    if False:
        for i in range(10):
            print('nop')
    max_feature_token_length = _get_text_feature_max_length(config, training_set_metadata)
    memory_usage = memory_usage / AUTOML_TEXT_ENCODER_MAX_TOKEN_LEN * max_feature_token_length
    return memory_usage

def compute_memory_usage(config_obj, training_set_metadata, model_category) -> int:
    if False:
        for i in range(10):
            print('nop')
    update_config_with_metadata(config_obj, training_set_metadata)
    lm = LudwigModel.create_model(config_obj)
    model_size = lm.get_model_size()
    batch_size = config_obj.trainer.batch_size
    if batch_size == AUTO:
        batch_size = MINIMUM_BATCH_SIZE
    memory_usage = model_size * (BYTES_PER_WEIGHT + BYTES_OPTIMIZER_PER_WEIGHT) * batch_size
    if model_category == TEXT:
        return _get_text_model_memory_usage(config_obj.to_dict(), training_set_metadata, memory_usage)
    else:
        return memory_usage

def sub_new_params(config: dict, new_param_vals: dict):
    if False:
        print('Hello World!')
    new_config = copy.deepcopy(config)
    for (param, val) in new_param_vals.items():
        config_section = param.split('.')[0]
        param_name = param.split('.')[1]
        new_config[config_section][param_name] = val
    return new_config

def get_new_params(current_param_values, hyperparam_search_space, params_to_modify):
    if False:
        i = 10
        return i + 15
    for (param, _) in params_to_modify.items():
        if param in hyperparam_search_space:
            if hyperparam_search_space[param][SPACE] == 'choice':
                current_param_values[param] = hyperparam_search_space[param]['categories'][-1]
            else:
                current_param_values[param] = hyperparam_search_space[param]['upper']
    return current_param_values

def _update_text_encoder(input_features: List, old_text_encoder: str, new_text_encoder: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    for feature in input_features:
        if feature['type'] == TEXT and feature['encoder'] == old_text_encoder:
            feature['encoder'] = new_text_encoder

def _get_text_feature_min_usable_length(input_features: List, training_set_metadata) -> int:
    if False:
        return 10
    'Returns min of AUTOML_SMALLER_TEXT_LENGTH and lowest 99th percentile sequence length over text features.'
    min_usable_length = AUTOML_SMALLER_TEXT_LENGTH
    for feature in input_features:
        if feature['type'] == TEXT:
            feature_99ptile_len = training_set_metadata[feature['name']]['max_sequence_length_99ptile']
            if feature_99ptile_len < min_usable_length:
                min_usable_length = feature_99ptile_len
    return round(min_usable_length)

def reduce_text_feature_max_length(config, training_set_metadata) -> bool:
    if False:
        while True:
            i = 10
    'Reduce max sequence length, when viable, to control its quadratic impact.'
    input_features = config['input_features']
    min_usable_length = _get_text_feature_min_usable_length(input_features, training_set_metadata)
    seq_len_limit = {'max_sequence_length': min_usable_length}
    if 'preprocessing' not in config:
        config['preprocessing'] = {TEXT: seq_len_limit}
    elif TEXT not in config['preprocessing'] or 'max_sequence_length' not in config['preprocessing'][TEXT] or min_usable_length < float(config['preprocessing'][TEXT]['max_sequence_length']):
        config['preprocessing'][TEXT] = seq_len_limit
    else:
        return False
    return True

def _update_num_samples(num_samples, hyperparam_search_space):
    if False:
        while True:
            i = 10
    max_num_samples = 1
    for param in hyperparam_search_space.keys():
        if hyperparam_search_space[param][SPACE] == 'choice':
            max_num_samples *= len(hyperparam_search_space[param]['categories'])
        else:
            return num_samples
    if max_num_samples < num_samples:
        return max_num_samples
    return num_samples

def memory_tune_config(config, dataset, model_category, row_count, backend):
    if False:
        return 10
    backend = initialize_backend(backend)
    fits_in_memory = False
    tried_reduce_seq_len = False
    config_obj = ModelConfig.from_dict(config)
    raw_config = config_obj.to_dict()
    training_set_metadata = get_trainingset_metadata(raw_config, dataset, backend)
    modified_hyperparam_search_space = copy.deepcopy(raw_config[HYPEROPT]['parameters'])
    current_param_values = {}
    param_list = []
    model_type = get_model_type(raw_config)
    if model_type in RANKED_MODIFIABLE_PARAM_LIST:
        params_to_modify = RANKED_MODIFIABLE_PARAM_LIST[model_type]
        if len(params_to_modify.keys()) > 0:
            param_list = list(params_to_modify.keys())
            max_memory = _get_machine_memory()
            initialize_pytorch()
    while param_list:
        current_param_values = get_new_params(current_param_values, modified_hyperparam_search_space, params_to_modify)
        temp_config = sub_new_params(raw_config, current_param_values)
        config_obj = ModelConfig.from_dict(temp_config)
        mem_use = compute_memory_usage(config_obj, training_set_metadata, model_category)
        if mem_use > max_memory and model_category == TEXT and (not tried_reduce_seq_len):
            tried_reduce_seq_len = True
            if reduce_text_feature_max_length(config, training_set_metadata):
                reduce_text_feature_max_length(temp_config, training_set_metadata)
                config_obj = ModelConfig.from_dict(temp_config)
                mem_use = compute_memory_usage(config_obj, training_set_metadata, model_category)
        logger.info(f'Checking model estimated mem use {mem_use} against memory size {max_memory}')
        if mem_use <= max_memory:
            fits_in_memory = True
            break
        (param, min_value) = (param_list[0], params_to_modify[param_list[0]])
        if param in modified_hyperparam_search_space.keys():
            param_space = modified_hyperparam_search_space[param]['space']
            if param_space == 'choice':
                if len(modified_hyperparam_search_space[param]['categories']) >= 2 and modified_hyperparam_search_space[param]['categories'][-2] >= min_value:
                    modified_hyperparam_search_space[param]['categories'] = modified_hyperparam_search_space[param]['categories'][:-1]
                else:
                    param_list.pop(0)
            else:
                (upper_bound, lower_bound) = (modified_hyperparam_search_space[param]['upper'], modified_hyperparam_search_space[param]['lower'])
                reduction_val = (upper_bound - lower_bound) * 0.1
                new_upper_bound = upper_bound - reduction_val
                if new_upper_bound > lower_bound and new_upper_bound > min_value:
                    modified_hyperparam_search_space[param]['upper'] = new_upper_bound
                else:
                    param_list.pop(0)
        else:
            param_list.pop(0)
    if model_category == TEXT and row_count > AUTOML_LARGE_TEXT_DATASET:
        if 'checkpoints_per_epoch' not in config[TRAINER] and 'steps_per_checkpoint' not in config[TRAINER]:
            checkpoints_per_epoch = max(2, math.floor(row_count / AUTOML_MAX_ROWS_PER_CHECKPOINT))
            config[TRAINER]['checkpoints_per_epoch'] = checkpoints_per_epoch
        if 'evaluate_training_set' not in config[TRAINER]:
            config[TRAINER]['evaluate_training_set'] = False
        if not fits_in_memory:
            _update_text_encoder(config['input_features'], AUTOML_DEFAULT_TEXT_ENCODER, AUTOML_SMALLER_TEXT_ENCODER)
    modified_config = copy.deepcopy(config)
    modified_config[HYPEROPT]['parameters'] = modified_hyperparam_search_space
    modified_config[HYPEROPT]['executor']['num_samples'] = _update_num_samples(modified_config[HYPEROPT]['executor']['num_samples'], modified_hyperparam_search_space)
    return (modified_config, fits_in_memory)