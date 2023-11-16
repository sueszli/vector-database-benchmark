"""TF general model utils."""
from __future__ import annotations
import functools
import gc
import inspect
import json
import os
import pickle
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import h5py
import numpy as np
import tensorflow as tf
from huggingface_hub import Repository, list_repo_files
from keras import backend as K
from packaging.version import parse
from tensorflow.python.util.keras_deps import get_call_context_function
from . import DataCollatorWithPadding, DefaultDataCollator
from .activations_tf import get_tf_activation
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import GenerationConfig, TFGenerationMixin
from .tf_utils import expand_1d, load_attributes_from_hdf5_group, save_attributes_to_hdf5_group, shape_list
from .utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, TF2_WEIGHTS_INDEX_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME, ModelOutput, PushToHubMixin, cached_file, download_url, find_labels, has_file, is_offline_mode, is_remote_url, is_safetensors_available, is_tf_symbolic_tensor, logging, requires_backends, working_or_temp_dir
from .utils.hub import convert_file_size_to_int, get_checkpoint_shard_files
if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.tensorflow import save_file as safe_save_file
if TYPE_CHECKING:
    from . import PreTrainedTokenizerBase
logger = logging.get_logger(__name__)
tf_logger = tf.get_logger()
TFModelInputType = Union[List[tf.Tensor], List[np.ndarray], Dict[str, tf.Tensor], Dict[str, np.ndarray], tf.Tensor, np.ndarray]

def dummy_loss(y_true, y_pred):
    if False:
        return 10
    if y_pred.shape.rank <= 1:
        return y_pred
    else:
        reduction_axes = list(range(1, y_pred.shape.rank))
        return tf.reduce_mean(y_pred, axis=reduction_axes)

class TFModelUtilsMixin:
    """
    A few utilities for `tf.keras.Model`, to be used as a mixin.
    """

    def num_parameters(self, only_trainable: bool=False) -> int:
        if False:
            while True:
                i = 10
        '\n        Get the number of (optionally, trainable) parameters in the model.\n\n        Args:\n            only_trainable (`bool`, *optional*, defaults to `False`):\n                Whether or not to return only the number of trainable parameters\n\n        Returns:\n            `int`: The number of parameters.\n        '
        if only_trainable:
            return int(sum((np.prod(w.shape.as_list()) for w in self.trainable_variables)))
        else:
            return self.count_params()

def keras_serializable(cls):
    if False:
        print('Hello World!')
    '\n    Decorate a Keras Layer class to support Keras serialization.\n\n    This is done by:\n\n    1. Adding a `transformers_config` dict to the Keras config dictionary in `get_config` (called by Keras at\n       serialization time.\n    2. Wrapping `__init__` to accept that `transformers_config` dict (passed by Keras at deserialization time) and\n       convert it to a config object for the actual layer initializer.\n    3. Registering the class as a custom object in Keras (if the Tensorflow version supports this), so that it does not\n       need to be supplied in `custom_objects` in the call to `tf.keras.models.load_model`.\n\n    Args:\n        cls (a `tf.keras.layers.Layers subclass`):\n            Typically a `TF.MainLayer` class in this project, in general must accept a `config` argument to its\n            initializer.\n\n    Returns:\n        The same class object, with modifications for Keras deserialization.\n    '
    initializer = cls.__init__
    config_class = getattr(cls, 'config_class', None)
    if config_class is None:
        raise AttributeError('Must set `config_class` to use @keras_serializable')

    @functools.wraps(initializer)
    def wrapped_init(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        config = args[0] if args and isinstance(args[0], PretrainedConfig) else kwargs.pop('config', None)
        if isinstance(config, dict):
            config = config_class.from_dict(config)
            initializer(self, config, *args, **kwargs)
        elif isinstance(config, PretrainedConfig):
            if len(args) > 0:
                initializer(self, *args, **kwargs)
            else:
                initializer(self, config, *args, **kwargs)
        else:
            raise ValueError('Must pass either `config` (PretrainedConfig) or `config` (dict)')
        self._config = config
        self._kwargs = kwargs
    cls.__init__ = wrapped_init
    if not hasattr(cls, 'get_config'):
        raise TypeError('Only use @keras_serializable on tf.keras.layers.Layer subclasses')
    if hasattr(cls.get_config, '_is_default'):

        def get_config(self):
            if False:
                return 10
            cfg = super(cls, self).get_config()
            cfg['config'] = self._config.to_dict()
            cfg.update(self._kwargs)
            return cfg
        cls.get_config = get_config
    cls._keras_serializable = True
    if hasattr(tf.keras.utils, 'register_keras_serializable'):
        cls = tf.keras.utils.register_keras_serializable()(cls)
    return cls

class TFCausalLanguageModelingLoss:
    """
    Loss function suitable for causal language modeling (CLM), that is, the task of guessing the next token.

    <Tip>

    Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    </Tip>
    """

    def hf_compute_loss(self, labels, logits):
        if False:
            return 10
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        if self.config.tf_legacy_loss:
            active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
            reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
            labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
            return loss_fn(labels, reduced_logits)
        unmasked_loss = loss_fn(tf.nn.relu(labels), logits)
        loss_mask = tf.cast(labels != -100, dtype=unmasked_loss.dtype)
        masked_loss = unmasked_loss * loss_mask
        reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask)
        return tf.reshape(reduced_masked_loss, (1,))

class TFQuestionAnsweringLoss:
    """
    Loss function suitable for question answering.
    """

    def hf_compute_loss(self, labels, logits):
        if False:
            return 10
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        start_loss = loss_fn(labels['start_position'], logits[0])
        end_loss = loss_fn(labels['end_position'], logits[1])
        return (start_loss + end_loss) / 2.0

class TFTokenClassificationLoss:
    """
    Loss function suitable for token classification.

    <Tip>

    Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    </Tip>
    """

    def hf_compute_loss(self, labels, logits):
        if False:
            while True:
                i = 10
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        if tf.executing_eagerly():
            if tf.math.reduce_any(labels == -1):
                tf.print('Using `-1` to mask the loss for the token is deprecated. Please use `-100` instead.')
        if self.config.tf_legacy_loss:
            if tf.math.reduce_any(labels == -1):
                tf.print('Using `-1` to mask the loss for the token is deprecated. Please use `-100` instead.')
                active_loss = tf.reshape(labels, (-1,)) != -1
            else:
                active_loss = tf.reshape(labels, (-1,)) != -100
            reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
            labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
            return loss_fn(labels, reduced_logits)
        unmasked_loss = loss_fn(tf.nn.relu(labels), logits)
        loss_mask = tf.cast(labels >= 0, dtype=unmasked_loss.dtype)
        masked_loss = unmasked_loss * loss_mask
        reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask)
        return tf.reshape(reduced_masked_loss, (1,))

class TFSequenceClassificationLoss:
    """
    Loss function suitable for sequence classification.
    """

    def hf_compute_loss(self, labels, logits):
        if False:
            i = 10
            return i + 15
        if logits.shape.rank == 1 or logits.shape[1] == 1:
            loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            if labels.shape.rank == 1:
                labels = tf.expand_dims(labels, axis=-1)
        else:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        return loss_fn(labels, logits)

class TFMultipleChoiceLoss:
    """Loss function suitable for multiple choice tasks."""

    def hf_compute_loss(self, labels, logits):
        if False:
            while True:
                i = 10
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        return loss_fn(labels, logits)

class TFMaskedLanguageModelingLoss(TFCausalLanguageModelingLoss):
    """
    Loss function suitable for masked language modeling (MLM), that is, the task of guessing the masked tokens.

    <Tip>

    Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    </Tip>
    """

class TFNextSentencePredictionLoss:
    """
    Loss function suitable for next sentence prediction (NSP), that is, the task of guessing the next sentence.

    <Tip>

    Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    </Tip>
    """

    def hf_compute_loss(self, labels, logits):
        if False:
            for i in range(10):
                print('nop')
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        if self.config.tf_legacy_loss:
            next_sentence_active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
            next_sentence_reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, 2)), next_sentence_active_loss)
            next_sentence_label = tf.boolean_mask(tf.reshape(labels, (-1,)), next_sentence_active_loss)
            return loss_fn(next_sentence_label, next_sentence_reduced_logits)
        unmasked_ns_loss = loss_fn(y_true=tf.nn.relu(labels), y_pred=logits)
        ns_loss_mask = tf.cast(labels != -100, dtype=unmasked_ns_loss.dtype)
        masked_ns_loss = unmasked_ns_loss * ns_loss_mask
        return masked_ns_loss

def booleans_processing(config, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Process the input booleans of each model.\n\n    Args:\n        config ([`PretrainedConfig`]):\n            The config of the running model.\n        **kwargs:\n            The boolean parameters\n\n    Returns:\n        A dictionary with the proper values for each boolean\n    '
    final_booleans = {}
    if 'output_attentions' in kwargs:
        final_booleans['output_attentions'] = kwargs['output_attentions'] if kwargs['output_attentions'] is not None else config.output_attentions
    final_booleans['output_hidden_states'] = kwargs['output_hidden_states'] if kwargs['output_hidden_states'] is not None else config.output_hidden_states
    final_booleans['return_dict'] = kwargs['return_dict'] if kwargs['return_dict'] is not None else config.return_dict
    if 'use_cache' in kwargs:
        final_booleans['use_cache'] = kwargs['use_cache'] if kwargs['use_cache'] is not None else getattr(config, 'use_cache', None)
    return final_booleans

def unpack_inputs(func):
    if False:
        return 10
    '\n    Decorator that processes the inputs to a Keras layer, passing them to the layer as keyword arguments. This enables\n    downstream use of the inputs by their variable name, even if they arrive packed as a dictionary in the first input\n    (common case in Keras).\n\n    Args:\n        func (`callable`):\n            The callable function of the TensorFlow model.\n\n\n    Returns:\n        A callable that wraps the original `func` with the behavior described above.\n    '
    original_signature = inspect.signature(func)

    @functools.wraps(func)
    def run_call_with_unpacked_inputs(self, *args, **kwargs):
        if False:
            return 10
        kwargs_call = {key: val for (key, val) in kwargs.items() if key not in dict(original_signature.parameters)}
        fn_args_and_kwargs = {key: val for (key, val) in kwargs.items() if key not in kwargs_call}
        fn_args_and_kwargs.update({'kwargs_call': kwargs_call})
        fn_args_and_kwargs.update(dict(zip(func.__code__.co_varnames[1:], args)))
        if 'EncoderDecoder' in self.__class__.__name__:
            config = None
        else:
            config = self.config
        unpacked_inputs = input_processing(func, config, **fn_args_and_kwargs)
        return func(self, **unpacked_inputs)
    run_call_with_unpacked_inputs.__signature__ = original_signature
    return run_call_with_unpacked_inputs

def input_processing(func, config, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Process the input of each TensorFlow model including the booleans. In case of a list of symbolic inputs, each input\n    has to be named accordingly to the parameters name, i.e. `input_ids = tf.keras.Input(shape=(128,), dtype=\'int32\',\n    name="input_ids")` otherwise the order of the tensors will not be guaranteed during the training.\n\n    Args:\n        func (`callable`):\n            The callable function of the TensorFlow model.\n        config ([`PretrainedConfig`]):\n            The config of the running model.\n        **kwargs:\n            The inputs of the model.\n\n    Returns:\n        Two lists, one for the missing layers, and another one for the unexpected layers.\n    '
    signature = dict(inspect.signature(func).parameters)
    has_kwargs = bool(signature.pop('kwargs', None))
    signature.pop('self', None)
    parameter_names = list(signature.keys())
    main_input_name = parameter_names[0]
    main_input = kwargs.pop(main_input_name, None)
    output = {}
    allowed_types = (tf.Tensor, bool, int, ModelOutput, tuple, list, dict, np.ndarray)
    if 'inputs' in kwargs['kwargs_call']:
        warnings.warn('The `inputs` argument is deprecated and will be removed in a future version, use `input_ids` instead.', FutureWarning)
        output['input_ids'] = kwargs['kwargs_call'].pop('inputs')
    if 'decoder_cached_states' in kwargs['kwargs_call']:
        warnings.warn('The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.', FutureWarning)
        output['past_key_values'] = kwargs['kwargs_call'].pop('decoder_cached_states')
    if 'past' in kwargs['kwargs_call'] and 'past_key_values' in parameter_names:
        warnings.warn('The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.', FutureWarning)
        kwargs['past_key_values'] = kwargs['kwargs_call'].pop('past')
    elif 'past_key_values' in kwargs['kwargs_call'] and 'past' in parameter_names:
        kwargs['past'] = kwargs['kwargs_call'].pop('past_key_values')
    if has_kwargs:
        output['kwargs'] = kwargs.pop('kwargs_call', {})
    else:
        if len(kwargs['kwargs_call']) > 0:
            raise ValueError(f"The following keyword arguments are not supported by this model: {list(kwargs['kwargs_call'].keys())}.")
        kwargs.pop('kwargs_call')
    for (k, v) in kwargs.items():
        if isinstance(v, allowed_types) or tf.is_tensor(v) or v is None:
            output[k] = v
        else:
            raise ValueError(f'Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.')
    if isinstance(main_input, (tuple, list)):
        for (i, input) in enumerate(main_input):
            if is_tf_symbolic_tensor(input):
                tensor_name = input.name.split(':')[0]
                if tensor_name in parameter_names:
                    output[tensor_name] = input
                else:
                    output[parameter_names[i]] = input
            elif isinstance(input, allowed_types) or input is None:
                output[parameter_names[i]] = input
            else:
                raise ValueError(f'Data of type {type(input)} is not allowed only {allowed_types} is accepted for {parameter_names[i]}.')
    elif isinstance(main_input, Mapping):
        if 'inputs' in main_input:
            warnings.warn('The `inputs` argument is deprecated and will be removed in a future version, use `input_ids` instead.', FutureWarning)
            output['input_ids'] = main_input.pop('inputs')
        if 'decoder_cached_states' in main_input:
            warnings.warn('The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.', FutureWarning)
            output['past_key_values'] = main_input.pop('decoder_cached_states')
        for (k, v) in dict(main_input).items():
            if isinstance(v, allowed_types) or v is None:
                output[k] = v
            elif k not in parameter_names and 'args' not in parameter_names:
                logger.warning(f'The parameter {k} does not belongs to the parameter list {parameter_names} and will be ignored.')
                continue
            else:
                raise ValueError(f'Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.')
    elif tf.is_tensor(main_input) or main_input is None:
        output[main_input_name] = main_input
    else:
        raise ValueError(f'Data of type {type(main_input)} is not allowed only {allowed_types} is accepted for {main_input_name}.')
    for name in parameter_names:
        if name not in list(output.keys()) and name != 'args':
            output[name] = kwargs.pop(name, signature[name].default)
    if 'args' in output:
        if output['args'] is not None and is_tf_symbolic_tensor(output['args']):
            tensor_name = output['args'].name.split(':')[0]
            output[tensor_name] = output['args']
        else:
            output['input_ids'] = output['args']
        del output['args']
    if 'kwargs' in output:
        del output['kwargs']
    cast_output = {}
    for (key, val) in output.items():
        if isinstance(val, tf.Tensor) and val.dtype == tf.int64:
            cast_output[key] = tf.cast(val, tf.int32)
        elif isinstance(val, np.ndarray) and val.dtype == np.int64:
            cast_output[key] = val.astype(np.int32)
        else:
            cast_output[key] = val
    output = cast_output
    del cast_output
    if config is not None:
        boolean_dict = {k: v for (k, v) in output.items() if k in ['return_dict', 'output_attentions', 'output_hidden_states', 'use_cache']}
        output.update(booleans_processing(config=config, **boolean_dict))
    return output

def dtype_byte_size(dtype):
    if False:
        print('Hello World!')
    '\n    Returns the size (in bytes) occupied by one parameter of type `dtype`.\n\n    Example:\n\n    ```py\n    >>> dtype_byte_size(tf.float32)\n    4\n    ```\n    '
    if dtype == tf.bool:
        return 1 / 8
    bit_search = re.search('[^\\d](\\d+)$', dtype.name)
    if bit_search is None:
        raise ValueError(f'`dtype` is not a valid dtype: {dtype}.')
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8

def strip_model_name_and_prefix(name, _prefix=None):
    if False:
        while True:
            i = 10
    if _prefix is not None and name.startswith(_prefix):
        name = name[len(_prefix):]
        if name.startswith('/'):
            name = name[1:]
    if 'model.' not in name and len(name.split('/')) > 1:
        name = '/'.join(name.split('/')[1:])
    return name

def tf_shard_checkpoint(weights, max_shard_size='10GB'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Splits a model state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a\n    given size.\n\n    The sub-checkpoints are determined by iterating through the `state_dict` in the order of its keys, so there is no\n    optimization made to make each sub-checkpoint as close as possible to the maximum size passed. For example, if the\n    limit is 10GB and we have weights of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as [6GB], [6+2GB],\n    [6+2+2GB] and not [6+2+2GB], [6+2GB], [6GB].\n\n    <Tip warning={true}>\n\n    If one of the model\'s weight is bigger that `max_shard_size`, it will end up in its own sub-checkpoint which will\n    have a size greater than `max_shard_size`.\n\n    </Tip>\n\n    Args:\n        weights (`Dict[str, tf.RessourceVariable]`): The list of tf.RessourceVariable of a model to save.\n        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):\n            The maximum size of each sub-checkpoint. If expressed as a string, needs to be digits followed by a unit\n            (like `"5MB"`).\n    '
    max_shard_size = convert_file_size_to_int(max_shard_size)
    sharded_state_dicts = []
    current_block = []
    current_block_size = 0
    total_size = 0
    for item in weights:
        weight_size = item.numpy().size * dtype_byte_size(item.dtype)
        if current_block_size + weight_size > max_shard_size:
            sharded_state_dicts.append(current_block)
            current_block = []
            current_block_size = 0
        current_block.append(item)
        current_block_size += weight_size
        total_size += weight_size
    sharded_state_dicts.append(current_block)
    if len(sharded_state_dicts) == 1:
        return ({TF2_WEIGHTS_NAME: sharded_state_dicts[0]}, None)
    weight_map = {}
    shards = {}
    for (idx, shard) in enumerate(sharded_state_dicts):
        shard_file = TF2_WEIGHTS_NAME.replace('.h5', f'-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.h5')
        shards[shard_file] = shard
        for weight in shard:
            weight_name = weight.name
            weight_map[weight_name] = shard_file
    metadata = {'total_size': total_size}
    index = {'metadata': metadata, 'weight_map': weight_map}
    return (shards, index)

def load_tf_sharded_weights(model, shard_files, ignore_mismatched_sizes=False, strict=False, _prefix=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    This is the same as `load_tf_weights` but for a sharded checkpoint. Detect missing and unexpected layers and load\n    the TF weights from the shard file accordingly to their names and shapes.\n\n    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being\n    loaded in the model.\n\n    Args:\n        model (`tf.keras.models.Model`): The model in which to load the checkpoint.\n        shard_files (`str` or `os.PathLike`): A list containing the sharded checkpoint names.\n        ignore_mismatched_sizes`bool`, *optional`, defaults to `True`):\n            Whether or not to ignore the mismatch between the sizes\n        strict (`bool`, *optional*, defaults to `True`):\n            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.\n\n    Returns:\n        Three lists, one for the missing layers, another one for the unexpected layers, and a last one for the\n        mismatched layers.\n    '
    unexpected_keys = set()
    saved_keys = set()
    mismatched_keys = set()
    model_keys = set()
    model_layer_map = {}
    for (i, k) in enumerate(model.weights):
        layer_name = k.name
        if _prefix is not None and layer_name.startswith(_prefix):
            layer_name = layer_name[len(_prefix):]
            layer_name = layer_name.lstrip('/')
        if not ('model.' in layer_name or len(layer_name.split('/')) == 1):
            layer_name = '/'.join(layer_name.split('/')[1:])
        model_keys.add(layer_name)
        model_layer_map[layer_name] = i
    for shard_file in shard_files:
        (saved_weight_names_set, unexpected_keys_set, mismatched_keys_set) = load_tf_shard(model, model_layer_map, shard_file, ignore_mismatched_sizes=ignore_mismatched_sizes, _prefix=_prefix)
        saved_keys.update(saved_weight_names_set)
        unexpected_keys.update(unexpected_keys_set)
        mismatched_keys.update(mismatched_keys_set)
        gc.collect()
    missing_keys = model_keys - saved_keys
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = f'Error(s) in loading state_dict for {model.__class__.__name__}'
        if len(missing_keys) > 0:
            str_missing_keys = ','.join([f'"{k}"' for k in missing_keys])
            error_message += f'\nMissing key(s): {str_missing_keys}.'
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ','.join([f'"{k}"' for k in unexpected_keys])
            error_message += f'\nMissing key(s): {str_unexpected_keys}.'
        raise RuntimeError(error_message)
    return (missing_keys, unexpected_keys, mismatched_keys)

def load_tf_shard(model, model_layer_map, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Loads a shard from a sharded checkpoint file. Handles the missing keys and unexpected keys.\n\n    Args:\n        model (`tf.keras.models.Model`): Model in which the weights are loaded\n        model_layer_map (`Dict`): A dictionary mapping the layer name to the index of the layer in the model.\n        resolved_archive_file (`str`): Path to the checkpoint file from which the weights will be loaded\n        ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`): Whether to ignore the mismatched keys\n\n    Returns:\n        `tf.keras.models.Model`: Three lists, one for the layers that were found and succesfully restored (from the\n        shard file), one for the mismatched layers, and another one for the unexpected layers.\n    '
    saved_weight_names_set = set()
    saved_weights = {}
    mismatched_keys = set()
    unexpected_keys = set()
    try:
        with h5py.File(resolved_archive_file, 'r') as sharded_checkpoint_file:
            saved_h5_model_layers_name = set(load_attributes_from_hdf5_group(sharded_checkpoint_file, 'layer_names'))
            weight_value_tuples = []
            for layer_name in saved_h5_model_layers_name:
                h5_layer_object = sharded_checkpoint_file[layer_name]
                saved_weights[layer_name] = np.asarray(h5_layer_object)
                saved_weight_names_set.add(layer_name)
                if layer_name not in model_layer_map:
                    unexpected_keys.add(layer_name)
                else:
                    symbolic_weight = model.weights[model_layer_map[layer_name]]
                    saved_weight_value = saved_weights[layer_name]
                    if saved_weight_value is not None:
                        if K.int_shape(symbolic_weight) != saved_weight_value.shape:
                            try:
                                array = np.reshape(saved_weight_value, K.int_shape(symbolic_weight))
                            except ValueError as e:
                                if ignore_mismatched_sizes:
                                    mismatched_keys.add((layer_name, saved_weight_value.shape, K.int_shape(symbolic_weight)))
                                    continue
                                else:
                                    raise e
                        else:
                            array = saved_weight_value
                    weight_value_tuples.append((symbolic_weight, array))
        K.batch_set_value(weight_value_tuples)
        return (saved_weight_names_set, unexpected_keys, mismatched_keys)
    except Exception as e:
        try:
            with open(resolved_archive_file) as f:
                if f.read().startswith('version'):
                    raise OSError('You seem to have cloned a repository without having git-lfs installed. Please install git-lfs and run `git lfs install` followed by `git lfs pull` in the folder you cloned.')
                else:
                    raise ValueError(f'Unable to locate the file {resolved_archive_file} which is necessary to load this pretrained model. Make sure you have saved the model properly.') from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(f"Unable to load weights from TF checkpoint file for '{resolved_archive_file}' at '{resolved_archive_file}'. If you tried to load a TF model from a sharded checkpoint, you should try converting the model by loading it in pytorch and saving it localy. A convertion script should be realeased soon.")

def load_tf_weights(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    if False:
        i = 10
        return i + 15
    "\n    Detect missing and unexpected layers and load the TF weights from the shard file accordingly to their names and\n    shapes.\n\n    Args:\n        model (`tf.keras.models.Model`):\n            The model to load the weights into.\n        resolved_archive_file (`str`):\n            The location of the H5 file.\n        ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):\n            Whether or not to ignore weights with shapes that don't match between the checkpoint of the model.\n\n    Returns:\n        Three lists, one for the missing layers, another one for the unexpected layers, and a last one for the\n        mismatched layers.\n    "
    if resolved_archive_file.endswith('.safetensors'):
        load_function = load_tf_weights_from_safetensors
    else:
        load_function = load_tf_weights_from_h5
    return load_function(model, resolved_archive_file, ignore_mismatched_sizes=ignore_mismatched_sizes, _prefix=_prefix)

def load_tf_weights_from_h5(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    if False:
        return 10
    mismatched_layers = []
    with h5py.File(resolved_archive_file, 'r') as sharded_checkpoint_file:
        saved_h5_model_layers_name = set(load_attributes_from_hdf5_group(sharded_checkpoint_file, 'layer_names'))
        missing_layers = list({layer.name for layer in model.layers} - saved_h5_model_layers_name)
        unexpected_layers = list(saved_h5_model_layers_name - {layer.name for layer in model.layers})
        saved_weight_names_set = set()
        symbolic_weights_names = set()
        weight_value_tuples = []
        for layer in model.layers:
            if layer.name in saved_h5_model_layers_name:
                h5_layer_object = sharded_checkpoint_file[layer.name]
                symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                saved_weights = {}
                for weight_name in load_attributes_from_hdf5_group(h5_layer_object, 'weight_names'):
                    name = '/'.join(weight_name.split('/')[1:])
                    if _prefix is not None:
                        name = _prefix + '/' + name
                    saved_weights[name] = np.asarray(h5_layer_object[weight_name])
                    saved_weight_names_set.add(name)
                for symbolic_weight in symbolic_weights:
                    if _prefix is not None:
                        delimeter = len(_prefix.split('/'))
                        symbolic_weight_name = '/'.join(symbolic_weight.name.split('/')[:delimeter] + symbolic_weight.name.split('/')[delimeter + 1:])
                    else:
                        symbolic_weight_name = '/'.join(symbolic_weight.name.split('/')[1:])
                    saved_weight_value = saved_weights.get(symbolic_weight_name, None)
                    if saved_weight_value is None and symbolic_weight_name.endswith('embeddings:0'):
                        symbolic_weight_name = symbolic_weight_name[:-12] + 'weight:0'
                        saved_weight_value = saved_weights.get(symbolic_weight_name, None)
                    symbolic_weights_names.add(symbolic_weight_name)
                    if saved_weight_value is not None:
                        if K.int_shape(symbolic_weight) != saved_weight_value.shape:
                            try:
                                array = np.reshape(saved_weight_value, K.int_shape(symbolic_weight))
                            except ValueError as e:
                                if ignore_mismatched_sizes:
                                    mismatched_layers.append((symbolic_weight_name, saved_weight_value.shape, K.int_shape(symbolic_weight)))
                                    continue
                                else:
                                    raise e
                        else:
                            array = saved_weight_value
                        weight_value_tuples.append((symbolic_weight, array))
    K.batch_set_value(weight_value_tuples)
    missing_layers.extend(list(symbolic_weights_names - saved_weight_names_set))
    unexpected_layers.extend(list(saved_weight_names_set - symbolic_weights_names))
    return (missing_layers, unexpected_layers, mismatched_layers)

def load_tf_weights_from_safetensors(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    if False:
        i = 10
        return i + 15
    with safe_open(resolved_archive_file, framework='tf') as safetensors_archive:
        mismatched_layers = []
        weight_names = [strip_model_name_and_prefix(w.name, _prefix=_prefix) for w in model.weights]
        loaded_weight_names = list(safetensors_archive.keys())
        missing_layers = list(set(weight_names) - set(loaded_weight_names))
        unexpected_layers = list(set(loaded_weight_names) - set(weight_names))
        for weight in model.weights:
            weight_name = strip_model_name_and_prefix(weight.name, _prefix=_prefix)
            if weight_name in loaded_weight_names:
                weight_value = safetensors_archive.get_tensor(weight_name)
                if K.int_shape(weight) != weight_value.shape:
                    try:
                        weight_value = tf.reshape(weight_value, K.int_shape(weight))
                    except (ValueError, tf.errors.InvalidArgumentError) as e:
                        if ignore_mismatched_sizes:
                            mismatched_layers.append((weight_name, weight_value.shape, K.int_shape(weight)))
                            continue
                        else:
                            raise e
                K.set_value(weight, weight_value)
    return (missing_layers, unexpected_layers, mismatched_layers)

def init_copy_embeddings(old_embeddings, new_num_tokens):
    if False:
        while True:
            i = 10
    '\n    This function aims to reduce the embeddings in case new_num_tokens < old_num_tokens or to pad with -1 in case\n    new_num_tokens > old_num_tokens. A mask is also computed in order to know which weight in the embeddings should be\n    kept or not. Example:\n\n        - if new_num_tokens=5 and old_num_tokens=4 and old_embeddings=[w1,w2,w3,w4]\n\n            -  mask=[True,True,True,True,False] and current_weights=[w1,w2,w3,w4,-1]\n        - if new_num_tokens=4 and old_num_tokens=5 and old_embeddings=[w1,w2,w3,w4,w5]\n\n            - mask=[True,True,True,True] and current_weights=[w1,w2,w3,w4]\n    '
    (old_num_tokens, old_embedding_dim) = shape_list(old_embeddings)
    size_diff = new_num_tokens - old_num_tokens
    if tf.math.greater(size_diff, 0):
        current_weights = tf.pad(old_embeddings.value(), tf.convert_to_tensor([[0, size_diff], [0, 0]]), constant_values=-1)
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        mask = tf.fill(tf.convert_to_tensor([num_tokens_to_copy, 1]), True)
        mask = tf.pad(mask, tf.convert_to_tensor([[0, size_diff], [0, 0]]), constant_values=False)
    else:
        current_weights = tf.slice(old_embeddings.value(), tf.convert_to_tensor([0, 0]), tf.convert_to_tensor([new_num_tokens, old_embedding_dim]))
        mask = tf.fill(tf.convert_to_tensor([new_num_tokens, 1]), True)
    return (mask, current_weights)

class TFPreTrainedModel(tf.keras.Model, TFModelUtilsMixin, TFGenerationMixin, PushToHubMixin):
    """
    Base class for all TF models.

    [`TFPreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models as well as a few methods common to all models to:

        - resize the input embeddings,
        - prune heads in the self-attention heads.

    Class attributes (overridden by derived classes):

        - **config_class** ([`PretrainedConfig`]) -- A subclass of [`PretrainedConfig`] to use as configuration class
          for this model architecture.
        - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
          classes of the same architecture adding modules on top of the base model.
        - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
          models, `pixel_values` for vision models and `input_values` for speech models).
    """
    config_class = None
    base_model_prefix = ''
    main_input_name = 'input_ids'
    _auto_class = None
    _using_dummy_loss = None
    _label_to_output_map = None
    _keys_to_ignore_on_load_missing = None
    _keys_to_ignore_on_load_unexpected = None
    _requires_load_weight_prefix = False

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        if False:
            return 10
        '\n        Dummy inputs to build the network.\n\n        Returns:\n            `Dict[str, tf.Tensor]`: The dummy inputs.\n        '
        dummies = {}
        for (key, spec) in self.input_signature.items():
            dummy_shape = [dim if dim is not None else 2 for dim in spec.shape]
            if spec.shape[0] is None:
                dummy_shape[0] = 1
            dummies[key] = tf.ones(shape=dummy_shape, dtype=spec.dtype)
            if key == 'token_type_ids':
                dummies[key] = tf.zeros_like(dummies[key])
        if self.config.add_cross_attention and 'encoder_hidden_states' in inspect.signature(self.call).parameters:
            if 'encoder_hidden_states' not in dummies:
                if self.main_input_name == 'input_ids':
                    dummies['encoder_hidden_states'] = tf.ones(shape=(1, 2, self.config.hidden_size), dtype=tf.float32, name='encoder_hidden_states')
                else:
                    raise NotImplementedError("Model has cross-attention but we couldn't infer the shape for the encoder hidden states. Please manually override dummy_inputs!")
        return dummies

    @property
    def framework(self) -> str:
        if False:
            return 10
        '\n        :str: Identifies that this is a TensorFlow model.\n        '
        return 'tf'

    def build(self, input_shape=None):
        if False:
            for i in range(10):
                print('nop')
        call_context = get_call_context_function()
        if self.built or call_context().in_call:
            self.built = True
        else:
            self.built = True
            self._set_save_spec(self.input_signature)
            self(self.dummy_inputs, training=False)

    def __init__(self, config, *inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*inputs, **kwargs)
        if not isinstance(config, PretrainedConfig):
            raise ValueError(f'Parameter config in `{self.__class__.__name__}(config)` should be an instance of class `PretrainedConfig`. To create a model from a pretrained model use `model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`')
        self.config = config
        self.name_or_path = config.name_or_path
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None

    def get_config(self):
        if False:
            while True:
                i = 10
        return self.config.to_dict()

    @classmethod
    def from_config(cls, config, **kwargs):
        if False:
            print('Hello World!')
        if isinstance(config, PretrainedConfig):
            return cls._from_config(config, **kwargs)
        return cls._from_config(cls.config_class.from_dict(config, **kwargs))

    @classmethod
    def _from_config(cls, config, **kwargs):
        if False:
            while True:
                i = 10
        '\n        All context managers that the model should be initialized under go here.\n        '
        return cls(config, **kwargs)

    def get_head_mask(self, head_mask: tf.Tensor | None, num_hidden_layers: int) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        Prepare the head mask if needed.\n\n        Args:\n            head_mask (`tf.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):\n                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).\n            num_hidden_layers (`int`):\n                The number of hidden layers in the model.\n\n        Returns:\n            `tf.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with\n            `[None]` for each layer.\n        '
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        if False:
            for i in range(10):
                print('nop')
        '-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]'
        if head_mask.shape.rank == 1:
            head_mask = head_mask[None, None, :, None, None]
            head_mask = tf.repeat(head_mask, repeats=num_hidden_layers, axis=0)
        elif head_mask.shape.rank == 2:
            head_mask = head_mask[:, None, :, None, None]
        assert head_mask.shape.rank == 5, f'head_mask.dim != 5, instead {head_mask.dim()}'
        head_mask = tf.cast(head_mask, tf.float32)
        return head_mask

    @tf.function
    def serving(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n        Method used for serving the model. Does not have a specific signature, but will be specialized as concrete\n        functions when saving with `save_pretrained`.\n            inputs (`Dict[str, tf.Tensor]`):\n                The input of the saved model as a dictionary of tensors.\n        '
        output = self.call(inputs)
        return self.serving_output(output)

    def eager_serving(self, inputs):
        if False:
            return 10
        '\n        Method used for serving the model. This method is deprecated, and will be removed.\n\n        Args:\n            inputs (`Dict[str, tf.Tensor]`):\n                The input of the saved model as a dictionary of tensors.\n        '
        warnings.warn('The function `eager_serving` is deprecated and will be removed in version 4.32.0 of Transformers', FutureWarning)
        output = self.call(inputs)
        return self.serving_output(output)

    @property
    def input_signature(self) -> Dict[str, tf.TensorSpec]:
        if False:
            i = 10
            return i + 15
        '\n        This property should return a dict mapping input names to tf.TensorSpec objects, representing the expected\n        shape and dtype for model inputs. It is used for both serving and for generating the dummy inputs used to build\n        the model.\n        '
        model_inputs = list(inspect.signature(self.call).parameters)
        sig = {}
        if 'input_ids' in model_inputs:
            if self.__class__.__name__.endswith('ForMultipleChoice'):
                text_dims = 3
            else:
                text_dims = 2
            for input_name in ('input_ids', 'attention_mask', 'token_type_ids', 'decoder_input_ids', 'decoder_attention_mask'):
                if input_name in model_inputs:
                    sig[input_name] = tf.TensorSpec([None] * text_dims, tf.int32, name=input_name)
        if 'pixel_values' in model_inputs:
            pixel_values_shape = [None, None, None, None]
            if hasattr(self.config, 'vision_config'):
                vision_config = self.config.vision_config
            else:
                vision_config = self.config
            if hasattr(vision_config, 'num_channels'):
                pixel_values_shape[1] = vision_config.num_channels
            else:
                raise NotImplementedError('Could not infer number of channels from config, please override input_signature to specify input shapes.')
            if hasattr(vision_config, 'image_size'):
                pixel_values_shape[2] = pixel_values_shape[3] = vision_config.image_size
            elif hasattr(vision_config, 'input_size'):
                pixel_values_shape[2] = pixel_values_shape[3] = vision_config.input_size
            else:
                raise NotImplementedError('Could not infer input image shape from config, please override input_signature to specify input shapes.')
            sig['pixel_values'] = tf.TensorSpec(pixel_values_shape, tf.float32, name='pixel_values')
        if 'input_features' in model_inputs:
            raise NotImplementedError('Audio models need a manually defined input_signature')
        return sig

    def serving_output(self, output):
        if False:
            while True:
                i = 10
        '\n        Prepare the output of the saved model. Can be overridden if specific serving modifications are required.\n        '
        if not isinstance(output, ModelOutput):
            return output
        for key in output:
            if key.endswith('hidden_states') and (not getattr(self.config, 'output_hidden_states', False)):
                output[key] = None
            elif key.endswith('attentions') and (not getattr(self.config, 'output_attentions', False)):
                output[key] = None
            elif key == 'past_key_values' and (not getattr(self.config, 'use_cache', False)):
                output[key] = None
            elif key == 'cross_attentions' and (not (getattr(self.config, 'output_attentions', False) and getattr(self.config, 'add_cross_attention', False))):
                output[key] = None
            if isinstance(output[key], (tuple, list)):
                try:
                    output[key] = tf.convert_to_tensor(output[key])
                except (ValueError, tf.errors.InvalidArgumentError):
                    pass
        return output

    @classmethod
    def can_generate(cls) -> bool:
        if False:
            print('Hello World!')
        '\n        Returns whether this model can generate sequences with `.generate()`.\n\n        Returns:\n            `bool`: Whether this model can generate sequences with `.generate()`.\n        '
        if 'GenerationMixin' in str(cls.prepare_inputs_for_generation) and 'GenerationMixin' in str(cls.generate):
            return False
        return True

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns the model's input embeddings layer.\n\n        Returns:\n            `tf.Variable`: The embeddings layer mapping vocabulary to hidden states.\n        "
        main_layer = getattr(self, self.base_model_prefix, self)
        if main_layer is not self:
            return main_layer.get_input_embeddings()
        else:
            raise NotImplementedError

    def _save_checkpoint(self, checkpoint_dir, epoch):
        if False:
            while True:
                i = 10
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        weights_path = os.path.join(checkpoint_dir, 'weights.h5')
        self.save_weights(weights_path)
        extra_data = {'epoch': epoch, 'optimizer_state': self.optimizer.get_weights()}
        extra_data_path = os.path.join(checkpoint_dir, 'extra_data.pickle')
        with open(extra_data_path, 'wb') as f:
            pickle.dump(extra_data, f)

    def load_repo_checkpoint(self, repo_path_or_name):
        if False:
            while True:
                i = 10
        '\n        Loads a saved checkpoint (model weights and optimizer state) from a repo. Returns the current epoch count when\n        the checkpoint was made.\n\n        Args:\n            repo_path_or_name (`str`):\n                Can either be a repository name for your {object} in the Hub or a path to a local folder (in which case\n                the repository will have the name of that local folder).\n\n        Returns:\n            `dict`: A dictionary of extra metadata from the checkpoint, most commonly an "epoch" count.\n        '
        if getattr(self, 'optimizer', None) is None:
            raise RuntimeError('Checkpoint loading failed as no optimizer is attached to the model. This is most likely caused by the model not being compiled.')
        if os.path.isdir(repo_path_or_name):
            local_dir = repo_path_or_name
        else:
            repo_files = list_repo_files(repo_path_or_name)
            for file in ('checkpoint/weights.h5', 'checkpoint/extra_data.pickle'):
                if file not in repo_files:
                    raise FileNotFoundError(f'Repo {repo_path_or_name} does not contain checkpoint file {file}!')
            repo = Repository(repo_path_or_name.split('/')[-1], clone_from=repo_path_or_name)
            local_dir = repo.local_dir
        checkpoint_dir = os.path.join(local_dir, 'checkpoint')
        weights_file = os.path.join(checkpoint_dir, 'weights.h5')
        if not os.path.isfile(weights_file):
            raise FileNotFoundError(f'Could not find checkpoint file weights.h5 in repo {repo_path_or_name}!')
        extra_data_file = os.path.join(checkpoint_dir, 'extra_data.pickle')
        if not os.path.isfile(extra_data_file):
            raise FileNotFoundError(f'Could not find checkpoint file extra_data.pickle in repo {repo_path_or_name}!')
        self.load_weights(weights_file)
        with open(extra_data_file, 'rb') as f:
            extra_data = pickle.load(f)
        self.optimizer.set_weights(extra_data['optimizer_state'])
        return {'epoch': extra_data['epoch']}

    def prepare_tf_dataset(self, dataset: 'datasets.Dataset', batch_size: int=8, shuffle: bool=True, tokenizer: Optional['PreTrainedTokenizerBase']=None, collate_fn: Optional[Callable]=None, collate_fn_args: Optional[Dict[str, Any]]=None, drop_remainder: Optional[bool]=None, prefetch: bool=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wraps a HuggingFace [`~datasets.Dataset`] as a `tf.data.Dataset` with collation and batching. This method is\n        designed to create a "ready-to-use" dataset that can be passed directly to Keras methods like `fit()` without\n        further modification. The method will drop columns from the dataset if they don\'t match input names for the\n        model. If you want to specify the column names to return rather than using the names that match this model, we\n        recommend using `Dataset.to_tf_dataset()` instead.\n\n        Args:\n            dataset (`Any`):\n                A [~`datasets.Dataset`] to be wrapped as a `tf.data.Dataset`.\n            batch_size (`int`, defaults to 8):\n                The size of batches to return.\n            shuffle (`bool`, defaults to `True`):\n                Whether to return samples from the dataset in random order. Usually `True` for training datasets and\n                `False` for validation/test datasets.\n            tokenizer ([`PreTrainedTokenizerBase`], *optional*):\n                A `PreTrainedTokenizer` that will be used to pad samples to create batches. Has no effect if a specific\n                `collate_fn` is passed instead.\n            collate_fn (`Callable`, *optional*):\n                A function that collates samples from the dataset into a single batch. Defaults to\n                `DefaultDataCollator` if no `tokenizer` is supplied or `DataCollatorWithPadding` if a `tokenizer` is\n                passed.\n            collate_fn_args (`Dict[str, Any]`, *optional*):\n                A dict of arguments to pass to the `collate_fn` alongside the list of samples.\n            drop_remainder (`bool`, *optional*):\n                Whether to drop the final batch, if the batch_size does not evenly divide the dataset length. Defaults\n                to the same setting as `shuffle`.\n            prefetch (`bool`, defaults to `True`):\n                Whether to add prefetching to the end of the `tf.data` pipeline. This is almost always beneficial for\n                performance, but can be disabled in edge cases.\n\n\n        Returns:\n            `Dataset`: A `tf.data.Dataset` which is ready to pass to the Keras API.\n        '
        requires_backends(self, ['datasets'])
        import datasets
        if collate_fn is None:
            if tokenizer is None:
                collate_fn = DefaultDataCollator(return_tensors='np')
            else:
                collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='np')
        if collate_fn_args is None:
            collate_fn_args = {}
        if not isinstance(dataset, datasets.Dataset):
            raise TypeError('Dataset argument should be a datasets.Dataset!')
        model_inputs = list(inspect.signature(self.call).parameters)
        model_labels = find_labels(self.__class__)
        if 'cols_to_retain' in list(inspect.signature(dataset._get_output_signature).parameters.keys()):
            (output_signature, _) = dataset._get_output_signature(dataset, batch_size=None, collate_fn=collate_fn, collate_fn_args=collate_fn_args, cols_to_retain=model_inputs)
        else:
            unwanted_columns = [feature for feature in dataset.features if feature not in model_inputs and feature not in ('label_ids', 'label')]
            dataset = dataset.remove_columns(unwanted_columns)
            (output_signature, _) = dataset._get_output_signature(dataset, batch_size=None, collate_fn=collate_fn, collate_fn_args=collate_fn_args)
        output_columns = list(output_signature.keys())
        feature_cols = [col for col in output_columns if col in model_inputs and col not in model_labels]
        label_cols = [col for col in output_columns if col in model_labels]
        feature_cols = feature_cols[0] if len(feature_cols) == 1 else feature_cols
        label_cols = label_cols[0] if len(label_cols) == 1 else label_cols
        if drop_remainder is None:
            drop_remainder = shuffle
        tf_dataset = dataset.to_tf_dataset(columns=feature_cols, label_cols=label_cols, batch_size=batch_size, shuffle=shuffle, drop_remainder=drop_remainder, collate_fn=collate_fn, collate_fn_args=collate_fn_args, prefetch=prefetch)
        return tf_dataset

    def compile(self, optimizer='rmsprop', loss='auto_with_warning', metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        if False:
            while True:
                i = 10
        "\n        This is a thin wrapper that sets the model's loss output head as the loss if the user does not specify a loss\n        function themselves.\n        "
        if loss in ('auto_with_warning', 'passthrough'):
            logger.info("No loss specified in compile() - the model's internal loss computation will be used as the loss. Don't panic - this is a common way to train TensorFlow models in Transformers! To disable this behaviour please pass a loss argument, or explicitly pass `loss=None` if you do not want your model to compute a loss. You can also specify `loss='auto'` to get the internal loss without printing this info string.")
            loss = 'auto'
        if loss == 'auto':
            loss = dummy_loss
            self._using_dummy_loss = True
        else:
            self._using_dummy_loss = False
        parent_args = list(inspect.signature(tf.keras.Model.compile).parameters.keys())
        if 'steps_per_execution' in parent_args:
            super().compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights, weighted_metrics=weighted_metrics, run_eagerly=run_eagerly, steps_per_execution=steps_per_execution, **kwargs)
        else:
            super().compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights, weighted_metrics=weighted_metrics, run_eagerly=run_eagerly, experimental_steps_per_execution=steps_per_execution, **kwargs)

    def compute_loss(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if hasattr(tf.keras.Model, 'compute_loss'):
            return super().compute_loss(*args, **kwargs)
        else:
            warnings.warn('The old compute_loss method is deprecated as it conflicts with the Keras compute_loss method added in TF 2.8. If you want the original HF compute_loss, please call hf_compute_loss() instead. From TF versions >= 2.8, or Transformers versions >= 5, calling compute_loss() will get the Keras method instead.', FutureWarning)
            return self.hf_compute_loss(*args, **kwargs)

    def get_label_to_output_name_mapping(self):
        if False:
            print('Hello World!')
        arg_names = list(inspect.signature(self.call).parameters)
        if self._label_to_output_map is not None:
            return self._label_to_output_map
        elif 'start_positions' in arg_names:
            return {'start_positions': 'start_logits', 'end_positions': 'end_logits'}
        elif 'sentence_order_label' in arg_names:
            return {'labels': 'prediction_logits', 'sentence_order_label': 'sop_logits'}
        elif 'next_sentence_label' in arg_names:
            return {'labels': 'prediction_logits', 'next_sentence_label': 'seq_relationship_logits'}
        elif 'mc_labels' in arg_names:
            return {'labels': 'logits', 'mc_labels': 'mc_logits'}
        else:
            return {}

    def train_step(self, data):
        if False:
            i = 10
            return i + 15
        "\n        A modification of Keras's default `train_step` that correctly handles matching outputs to labels for our models\n        and supports directly training on the loss output head. In addition, it ensures input keys are copied to the\n        labels where appropriate. It will also copy label keys into the input dict when using the dummy loss, to ensure\n        that they are available to the model during the forward pass.\n        "
        arg_names = list(inspect.signature(self.call).parameters)
        label_kwargs = find_labels(self.__class__)
        label_to_output = self.get_label_to_output_name_mapping()
        output_to_label = {val: key for (key, val) in label_to_output.items()}
        if not self._using_dummy_loss and parse(tf.__version__) < parse('2.11.0'):
            data = expand_1d(data)
        (x, y, sample_weight) = tf.keras.utils.unpack_x_y_sample_weight(data)
        if isinstance(x, dict):
            x = x.copy()
        if isinstance(y, dict):
            y = y.copy()
        if self._using_dummy_loss and y is not None:
            if len(label_kwargs) == 1 and isinstance(y, tf.Tensor):
                if isinstance(x, tf.Tensor):
                    x = {arg_names[0]: x}
                label_kwarg = next(iter(label_kwargs))
                if label_kwarg not in x:
                    x[label_kwarg] = y
            elif isinstance(y, dict):
                if isinstance(x, tf.Tensor):
                    x = {arg_names[0]: x}
                for (key, val) in y.items():
                    if key in arg_names and key not in x:
                        x[key] = val
                    elif output_to_label.get(key, None) in arg_names and key not in x:
                        x[output_to_label[key]] = val
        if y is None:
            y = {key: val for (key, val) in x.items() if key in label_kwargs}
            if not y and (not self._using_dummy_loss):
                raise ValueError('Could not find label column(s) in input dict and no separate labels were provided!')
        if isinstance(y, dict):
            y = {label_to_output.get(key, key): val for (key, val) in y.items()}
        with tf.GradientTape() as tape:
            if self._using_dummy_loss and 'return_loss' in arg_names:
                y_pred = self(x, training=True, return_loss=True)
            else:
                y_pred = self(x, training=True)
            if self._using_dummy_loss:
                loss = self.compiled_loss(y_pred.loss, y_pred.loss, sample_weight, regularization_losses=self.losses)
            else:
                loss = None
            if isinstance(y, dict) and len(y) == 1:
                if list(y.keys())[0] in y_pred.keys():
                    y_pred = y_pred[list(y.keys())[0]]
                elif list(y_pred.keys())[0] == 'loss':
                    y_pred = y_pred[1]
                else:
                    y_pred = y_pred[0]
                (_, y) = y.popitem()
            elif isinstance(y, dict):
                y_pred = {key: val for (key, val) in y_pred.items() if key in y}
            elif isinstance(y, tuple) or isinstance(y, list):
                if list(y_pred.keys())[0] == 'loss':
                    y_pred = y_pred.to_tuple()[1:]
                else:
                    y_pred = y_pred.to_tuple()
                y_pred = y_pred[:len(y)]
            elif list(y_pred.keys())[0] == 'loss':
                y_pred = y_pred[1]
            else:
                y_pred = y_pred[0]
            if loss is None:
                loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def test_step(self, data):
        if False:
            return 10
        "\n        A modification of Keras's default `train_step` that correctly handles matching outputs to labels for our models\n        and supports directly training on the loss output head. In addition, it ensures input keys are copied to the\n        labels where appropriate. It will also copy label keys into the input dict when using the dummy loss, to ensure\n        that they are available to the model during the forward pass.\n        "
        arg_names = list(inspect.signature(self.call).parameters)
        label_kwargs = find_labels(self.__class__)
        label_to_output = self.get_label_to_output_name_mapping()
        output_to_label = {val: key for (key, val) in label_to_output.items()}
        if not self._using_dummy_loss and parse(tf.__version__) < parse('2.11.0'):
            data = expand_1d(data)
        (x, y, sample_weight) = tf.keras.utils.unpack_x_y_sample_weight(data)
        if isinstance(x, dict):
            x = x.copy()
        if isinstance(y, dict):
            y = y.copy()
        if self._using_dummy_loss and y is not None:
            arg_names = list(inspect.signature(self.call).parameters)
            if len(label_kwargs) == 1 and isinstance(y, tf.Tensor):
                if isinstance(x, tf.Tensor):
                    x = {arg_names[0]: x}
                label_kwarg = next(iter(label_kwargs))
                if label_kwarg not in x:
                    x[label_kwarg] = y
            elif isinstance(y, dict):
                if isinstance(x, tf.Tensor):
                    x = {arg_names[0]: x}
                for (key, val) in y.items():
                    if key in arg_names and key not in x:
                        x[key] = val
                    elif output_to_label.get(key, None) in arg_names and key not in x:
                        x[output_to_label[key]] = val
        if y is None:
            y = {key: val for (key, val) in x.items() if key in label_kwargs}
            if not y and (not self._using_dummy_loss):
                raise ValueError('Could not find label column(s) in input dict and no separate labels were provided!')
        if isinstance(y, dict):
            y = {label_to_output.get(key, key): val for (key, val) in y.items()}
        if self._using_dummy_loss and 'return_loss' in arg_names:
            y_pred = self(x, return_loss=True, training=False)
        else:
            y_pred = self(x, training=False)
        if self._using_dummy_loss:
            loss = self.compiled_loss(y_pred.loss, y_pred.loss, sample_weight, regularization_losses=self.losses)
        else:
            loss = None
        if isinstance(y, dict) and len(y) == 1:
            if list(y.keys())[0] in y_pred.keys():
                y_pred = y_pred[list(y.keys())[0]]
            elif list(y_pred.keys())[0] == 'loss':
                y_pred = y_pred[1]
            else:
                y_pred = y_pred[0]
            (_, y) = y.popitem()
        elif isinstance(y, dict):
            y_pred = {key: val for (key, val) in y_pred.items() if key in y}
        elif isinstance(y, tuple) or isinstance(y, list):
            if list(y_pred.keys())[0] == 'loss':
                y_pred = y_pred.to_tuple()[1:]
            else:
                y_pred = y_pred.to_tuple()
            y_pred = y_pred[:len(y)]
        elif list(y_pred.keys())[0] == 'loss':
            y_pred = y_pred[1]
        else:
            y_pred = y_pred[0]
        if loss is None:
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def create_model_card(self, output_dir, model_name: str, language: Optional[str]=None, license: Optional[str]=None, tags: Optional[str]=None, finetuned_from: Optional[str]=None, tasks: Optional[str]=None, dataset_tags: Optional[Union[str, List[str]]]=None, dataset: Optional[Union[str, List[str]]]=None, dataset_args: Optional[Union[str, List[str]]]=None):
        if False:
            print('Hello World!')
        '\n        Creates a draft of a model card using the information available to the `Trainer`.\n\n        Args:\n            output_dir (`str` or `os.PathLike`):\n                The folder in which to create the model card.\n            model_name (`str`, *optional*):\n                The name of the model.\n            language (`str`, *optional*):\n                The language of the model (if applicable)\n            license (`str`, *optional*):\n                The license of the model. Will default to the license of the pretrained model used, if the original\n                model given to the `Trainer` comes from a repo on the Hub.\n            tags (`str` or `List[str]`, *optional*):\n                Some tags to be included in the metadata of the model card.\n            finetuned_from (`str`, *optional*):\n                The name of the model used to fine-tune this one (if applicable). Will default to the name of the repo\n                of the original model given to the `Trainer` (if it comes from the Hub).\n            tasks (`str` or `List[str]`, *optional*):\n                One or several task identifiers, to be included in the metadata of the model card.\n            dataset_tags (`str` or `List[str]`, *optional*):\n                One or several dataset tags, to be included in the metadata of the model card.\n            dataset (`str` or `List[str]`, *optional*):\n                One or several dataset identifiers, to be included in the metadata of the model card.\n            dataset_args (`str` or `List[str]`, *optional*):\n               One or several dataset arguments, to be included in the metadata of the model card.\n        '
        from .modelcard import TrainingSummary
        training_summary = TrainingSummary.from_keras(self, keras_history=self.history, language=language, license=license, tags=tags, model_name=model_name, finetuned_from=finetuned_from, tasks=tasks, dataset_tags=dataset_tags, dataset=dataset, dataset_args=dataset_args)
        model_card = training_summary.to_model_card()
        with open(os.path.join(output_dir, 'README.md'), 'w') as f:
            f.write(model_card)

    def set_input_embeddings(self, value):
        if False:
            for i in range(10):
                print('nop')
        "\n        Set model's input embeddings\n\n        Args:\n            value (`tf.Variable`):\n                The new weights mapping hidden states to vocabulary.\n        "
        main_layer = getattr(self, self.base_model_prefix)
        if main_layer is None:
            raise NotImplementedError('The model does not implements the base_model_prefix attribute.')
        try:
            main_layer.set_input_embeddings(value)
        except AttributeError:
            logger.info('Building the model')
            self.build()
            main_layer.set_input_embeddings(value)

    def get_output_embeddings(self) -> Union[None, tf.keras.layers.Layer]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns the model's output embeddings\n\n        Returns:\n            `tf.Variable`: The new weights mapping vocabulary to hidden states.\n        "
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()
            try:
                return lm_head.get_output_embeddings()
            except AttributeError:
                logger.info('Building the model')
                self.build()
                return lm_head().get_output_embeddings()
        return None

    def set_output_embeddings(self, value):
        if False:
            return 10
        "\n        Set model's output embeddings\n\n        Args:\n            value (`tf.Variable`):\n                The new weights mapping hidden states to vocabulary.\n        "
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()
            try:
                lm_head.set_output_embeddings(value)
            except AttributeError:
                logger.info('Building the model')
                self.build()
                lm_head.set_output_embeddings(value)

    def get_output_layer_with_bias(self) -> Union[None, tf.keras.layers.Layer]:
        if False:
            return 10
        '\n        Get the layer that handles a bias attribute in case the model has an LM head with weights tied to the\n        embeddings\n\n        Return:\n            `tf.keras.layers.Layer`: The layer that handles the bias, None if not an LM model.\n        '
        warnings.warn('The method get_output_layer_with_bias is deprecated. Please use `get_lm_head` instead.', FutureWarning)
        return self.get_lm_head()

    def get_prefix_bias_name(self) -> Union[None, str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the concatenated _prefix name of the bias from the model name to the parent layer\n\n        Return:\n            `str`: The _prefix name of the bias.\n        '
        warnings.warn('The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.', FutureWarning)
        return None

    def get_bias(self) -> Union[None, Dict[str, tf.Variable]]:
        if False:
            while True:
                i = 10
        '\n        Dict of bias attached to an LM head. The key represents the name of the bias attribute.\n\n        Return:\n            `tf.Variable`: The weights representing the bias, None if not an LM model.\n        '
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()
            try:
                return lm_head.get_bias()
            except AttributeError:
                self.build()
                return lm_head.get_bias()
        return None

    def set_bias(self, value):
        if False:
            i = 10
            return i + 15
        '\n        Set all the bias in the LM head.\n\n        Args:\n            value (`Dict[tf.Variable]`):\n                All the new bias attached to an LM head.\n        '
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()
            try:
                lm_head.set_bias(value)
            except AttributeError:
                self.build()
                lm_head.set_bias(value)

    def get_lm_head(self) -> tf.keras.layers.Layer:
        if False:
            return 10
        '\n        The LM Head layer. This method must be overwritten by all the models that have a lm head.\n\n        Return:\n            `tf.keras.layers.Layer`: The LM head layer if the model has one, None if not.\n        '
        return None

    def resize_token_embeddings(self, new_num_tokens: Optional[int]=None) -> Union[tf.keras.layers.Embedding, tf.Variable]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.\n\n        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.\n\n        Arguments:\n            new_num_tokens (`int`, *optional*):\n                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized\n                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just\n                returns a pointer to the input tokens without doing anything.\n\n        Return:\n            `tf.Variable` or `tf.keras.layers.Embedding`: Pointer to the input tokens of the model.\n        '
        if isinstance(self.get_input_embeddings(), tf.keras.layers.Embedding):
            return self._v2_resized_token_embeddings(new_num_tokens)
        if new_num_tokens is None or new_num_tokens == self.config.vocab_size:
            return self._get_word_embedding_weight(self.get_input_embeddings())
        model_embeds = self._resize_token_embeddings(new_num_tokens)
        self.config.vocab_size = new_num_tokens
        return model_embeds

    def _v2_resized_token_embeddings(self, new_num_tokens: Optional[int]=None) -> tf.keras.layers.Embedding:
        if False:
            return 10
        '\n        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.\n\n        Arguments:\n            new_num_tokens (`int`, *optional*):\n                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized\n                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just\n                returns a pointer to the input tokens without doing anything.\n\n        Return:\n            `tf.keras.layers.Embedding`: Pointer to the input tokens of the model.\n        '
        if new_num_tokens is None or new_num_tokens == self.config.vocab_size:
            return self.get_input_embeddings()
        model_embeds = self._v2_resize_token_embeddings(new_num_tokens)
        self.config.vocab_size = new_num_tokens
        return model_embeds

    def _get_word_embedding_weight(model, embedding_layer):
        if False:
            print('Hello World!')
        if isinstance(embedding_layer, tf.Tensor):
            return embedding_layer
        embeds = getattr(embedding_layer, 'weight', None)
        if embeds is not None:
            return embeds
        embeds = getattr(embedding_layer, 'decoder', None)
        if embeds is not None:
            return embeds
        model.build()
        embeds = getattr(embedding_layer, 'weight', None)
        if embeds is not None:
            return embeds
        embeds = getattr(embedding_layer, 'decoder', None)
        if embeds is not None:
            return embeds
        return None

    def _resize_token_embeddings(self, new_num_tokens):
        if False:
            for i in range(10):
                print('nop')
        old_embeddings = self._get_word_embedding_weight(self.get_input_embeddings())
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        if self.get_bias() is not None:
            old_lm_head_bias = self.get_bias()
            new_lm_head_bias = self._get_resized_lm_head_bias(old_lm_head_bias, new_num_tokens)
            self.set_bias(new_lm_head_bias)
        if self.get_output_embeddings() is not None:
            old_lm_head_decoder = self._get_word_embedding_weight(self.get_output_embeddings())
            new_lm_head_decoder = self._get_resized_lm_head_decoder(old_lm_head_decoder, new_num_tokens)
            self.set_output_embeddings(new_lm_head_decoder)
        self.set_input_embeddings(new_embeddings)
        return self.get_input_embeddings()

    def _v2_resize_token_embeddings(self, new_num_tokens):
        if False:
            while True:
                i = 10
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._v2_get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)
        if self.get_bias() is not None:
            old_lm_head_bias = self.get_bias()
            new_lm_head_bias = self._v2_get_resized_lm_head_bias(old_lm_head_bias, new_num_tokens)
            self.set_bias(new_lm_head_bias)
        tied_weights = self.get_input_embeddings() == self.get_output_embeddings()
        if self.get_output_embeddings() is not None and (not tied_weights):
            old_lm_head_decoder = self._get_word_embedding_weight(self.get_output_embeddings())
            new_lm_head_decoder = self._get_resized_lm_head_decoder(old_lm_head_decoder, new_num_tokens)
            self.set_output_embeddings(new_lm_head_decoder)
        return self.get_input_embeddings()

    def _get_resized_lm_head_bias(self, old_lm_head_bias, new_num_tokens):
        if False:
            return 10
        '\n        Build a resized bias from the old ones. Increasing the size will add newly initialized vectors at the end.\n        Reducing the size will remove vectors from the end\n\n        Args:\n            old_lm_head_bias (`tf.Variable`):\n                Old lm head bias to be resized.\n            new_num_tokens (`int`, *optional*):\n                New number of tokens in the linear matrix.\n\n                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove\n                vectors from the end. If not provided or `None`, just returns None\n\n        Return:\n            `tf.Variable`: Pointer to the resized bias.\n        '
        new_lm_head_bias = {}
        for (attr, weight) in old_lm_head_bias.items():
            (first_dim, old_num_tokens) = (None, shape_list(weight)[0]) if tf.rank(weight) == 1 else shape_list(weight)
            size_diff = new_num_tokens - old_num_tokens
            final_shape = [new_num_tokens] if first_dim is None else [first_dim, new_num_tokens]
            if tf.math.greater(size_diff, 0):
                padding_shape = [[0, size_diff]] if first_dim is None else [[0, 0], [0, size_diff]]
                current_bias = tf.pad(weight.value(), tf.convert_to_tensor(padding_shape), constant_values=-1)
                num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
                mask_shape = [num_tokens_to_copy] if first_dim is None else [1, num_tokens_to_copy]
                bias_mask = tf.fill(tf.convert_to_tensor(mask_shape), True)
                bias_mask = tf.pad(bias_mask, tf.convert_to_tensor(padding_shape), constant_values=False)
            else:
                slice_from = [0] if first_dim is None else [0, 0]
                current_bias = tf.slice(weight.value(), tf.convert_to_tensor(slice_from), tf.convert_to_tensor(final_shape))
                bias_mask = tf.fill(tf.convert_to_tensor(final_shape), True)
            new_bias = self.add_weight(shape=final_shape, initializer='zeros', trainable=True, name=weight.name.split(':')[0])
            init_bias = tf.where(bias_mask, current_bias, new_bias.value())
            new_bias.assign(init_bias)
            new_lm_head_bias[attr] = new_bias
        return new_lm_head_bias

    def _v2_get_resized_lm_head_bias(self, old_lm_head_bias: Dict[str, tf.Variable], new_num_tokens: int) -> Dict[str, tf.Tensor]:
        if False:
            i = 10
            return i + 15
        '\n        Build a resized bias from the old ones. Increasing the size will add newly initialized vectors at the end.\n        Reducing the size will remove vectors from the end\n\n        Args:\n            old_lm_head_bias (`Dict[str, tf.Variable]`):\n                Old lm head bias to be resized.\n            new_num_tokens (`int`):\n                New number of tokens in the linear matrix. Increasing the size will add newly initialized vectors at\n                the end. Reducing the size will remove vectors from the end.\n\n        Return:\n            `tf.Tensor`: Values for the resized bias.\n        '
        new_lm_head_bias = {}
        for (attr, weight) in old_lm_head_bias.items():
            (first_dim, old_num_tokens) = (None, shape_list(weight)[0]) if tf.rank(weight) == 1 else shape_list(weight)
            size_diff = new_num_tokens - old_num_tokens
            if old_num_tokens > new_num_tokens:
                new_bias = weight.value()[..., :new_num_tokens]
            else:
                padding_shape = [[0, size_diff]] if first_dim is None else [[0, 0], [0, size_diff]]
                new_bias = tf.pad(weight.value(), tf.convert_to_tensor(padding_shape))
            new_lm_head_bias[attr] = new_bias
        return new_lm_head_bias

    def _get_resized_lm_head_decoder(self, old_lm_head_decoder, new_num_tokens):
        if False:
            for i in range(10):
                print('nop')
        '\n        Build a resized decoder from the old ones. Increasing the size will add newly initialized vectors at the end.\n        Reducing the size will remove vectors from the end\n\n        Args:\n            old_lm_head_decoder (`tf.Variable`):\n                Old lm head decoder to be resized.\n            new_num_tokens (`int`, *optional*):\n                New number of tokens in the linear matrix.\n\n                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove\n                vectors from the end. If not provided or `None`, just returns None\n\n        Return:\n            `tf.Variable`: Pointer to the resized decoder or None if the output embeddings are different from the input\n            ones.\n        '
        new_lm_head_decoder = old_lm_head_decoder
        is_input_output_equals = tf.reduce_any(self._get_word_embedding_weight(self.get_input_embeddings()) == old_lm_head_decoder)
        if old_lm_head_decoder is not None and (not is_input_output_equals):
            old_embedding_dim = shape_list(old_lm_head_decoder)[1]
            (decoder_mask, current_decoder) = init_copy_embeddings(old_lm_head_decoder, new_num_tokens)
            new_lm_head_decoder = self.add_weight(shape=(new_num_tokens, old_embedding_dim), initializer='zeros', trainable=True, name=old_lm_head_decoder.name.split(':')[0])
            init_decoder = tf.where(decoder_mask, current_decoder, new_lm_head_decoder.value())
            new_lm_head_decoder.assign(init_decoder)
        return new_lm_head_decoder

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None) -> tf.Variable:
        if False:
            print('Hello World!')
        '\n        Build a resized Embedding weights from a provided token Embedding weights. Increasing the size will add newly\n        initialized vectors at the end. Reducing the size will remove vectors from the end\n\n        Args:\n            old_embeddings (`tf.Variable`):\n                Old embeddings to be resized.\n            new_num_tokens (`int`, *optional*):\n                New number of tokens in the embedding matrix.\n\n                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove\n                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens\n                `tf.Variable` module of the model without doing anything.\n\n        Return:\n            `tf.Variable`: Pointer to the resized Embedding Module or the old Embedding Module if `new_num_tokens` is\n            `None`\n        '
        old_embedding_dim = shape_list(old_embeddings)[1]
        init_range = getattr(self.config, 'initializer_range', 0.02)
        (embeddings_mask, current_embeddings) = init_copy_embeddings(old_embeddings, new_num_tokens)
        new_embeddings = self.add_weight(name=old_embeddings.name.split(':')[0], shape=[new_num_tokens, old_embedding_dim], initializer=get_initializer(init_range), dtype=tf.float32)
        init_embeddings = tf.where(embeddings_mask, current_embeddings, new_embeddings.value())
        new_embeddings.assign(init_embeddings)
        return new_embeddings

    def _v2_get_resized_embeddings(self, old_embeddings: tf.keras.layers.Embedding, new_num_tokens: int) -> tf.keras.layers.Embedding:
        if False:
            return 10
        '\n        Build a resized Embedding layer from a provided Embedding layer. Increasing the size will add newly initialized\n        vectors at the end. Reducing the size will remove vectors from the end.\n\n        Args:\n            old_embeddings (`tf.keras.layers.Embedding`):\n                Old embeddings to be resized.\n            new_num_tokens (`int`, *optional*):\n                New number of tokens in the embedding matrix.\n\n        Return:\n            `tf.keras.layers.Embedding`: Resized Embedding layer.\n        '
        init_range = 0.02
        potential_initialization_variable_names = ['initializer_range', 'initializer_factor', 'init_std']
        for var_name in potential_initialization_variable_names:
            if hasattr(self.config, var_name):
                init_range = getattr(self.config, var_name)
        new_embeddings = tf.keras.layers.Embedding(input_dim=new_num_tokens, output_dim=old_embeddings.output_dim, embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=init_range), name=old_embeddings.embeddings.name[:-13])
        new_embeddings(tf.constant([[0]]))
        if old_embeddings.input_dim >= new_num_tokens:
            init_embeddings = old_embeddings.embeddings[:new_num_tokens]
        else:
            init_embeddings = tf.concat([old_embeddings.embeddings, new_embeddings.embeddings[old_embeddings.input_dim:]], axis=0)
        new_embeddings.embeddings.assign(init_embeddings)
        return new_embeddings

    def prune_heads(self, heads_to_prune):
        if False:
            return 10
        '\n        Prunes heads of the base model.\n\n        Arguments:\n            heads_to_prune (`Dict[int, List[int]]`):\n                Dictionary with keys being selected layer indices (`int`) and associated values being the list of heads\n                to prune in said layer (list of `int`). For instance {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on\n                layer 1 and heads 2 and 3 on layer 2.\n        '
        raise NotImplementedError

    def save_pretrained(self, save_directory, saved_model=False, version=1, push_to_hub=False, signatures=None, max_shard_size: Union[int, str]='10GB', create_pr: bool=False, safe_serialization: bool=False, token: Optional[Union[str, bool]]=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Save a model and its configuration file to a directory, so that it can be re-loaded using the\n        [`~TFPreTrainedModel.from_pretrained`] class method.\n\n        Arguments:\n            save_directory (`str`):\n                Directory to which to save. Will be created if it doesn\'t exist.\n            saved_model (`bool`, *optional*, defaults to `False`):\n                If the model has to be saved in saved model format as well or not.\n            version (`int`, *optional*, defaults to 1):\n                The version of the saved model. A saved model needs to be versioned in order to be properly loaded by\n                TensorFlow Serving as detailed in the official documentation\n                https://www.tensorflow.org/tfx/serving/serving_basic\n            push_to_hub (`bool`, *optional*, defaults to `False`):\n                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the\n                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your\n                namespace).\n            signatures (`dict` or `tf.function`, *optional*):\n                Model\'s signature used for serving. This will be passed to the `signatures` argument of model.save().\n            max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):\n                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size\n                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).\n\n                <Tip warning={true}>\n\n                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard\n                which will be bigger than `max_shard_size`.\n\n                </Tip>\n\n            create_pr (`bool`, *optional*, defaults to `False`):\n                Whether or not to create a PR with the uploaded files or directly commit.\n            safe_serialization (`bool`, *optional*, defaults to `False`):\n                Whether to save the model using `safetensors` or the traditional TensorFlow way (that uses `h5`).\n            token (`str` or `bool`, *optional*):\n                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use\n                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).\n            kwargs (`Dict[str, Any]`, *optional*):\n                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.\n        '
        use_auth_token = kwargs.pop('use_auth_token', None)
        if use_auth_token is not None:
            warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
            if token is not None:
                raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
            token = use_auth_token
        if token is not None:
            kwargs['token'] = token
        if os.path.isfile(save_directory):
            logger.error(f'Provided path ({save_directory}) should be a directory, not a file')
            return
        os.makedirs(save_directory, exist_ok=True)
        if push_to_hub:
            commit_message = kwargs.pop('commit_message', None)
            repo_id = kwargs.pop('repo_id', save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)
        if saved_model:
            if getattr(self.config, 'torch_dtype', None) is not None and (not isinstance(self.config.torch_dtype, str)):
                self.config.torch_dtype = str(self.config.torch_dtype).split('.')[1]
            if signatures is None:
                serving_default = self.serving.get_concrete_function(self.input_signature)
                if any((spec.dtype == tf.int32 for spec in self.input_signature.values())):
                    int64_spec = {key: tf.TensorSpec(shape=spec.shape, dtype=tf.int64 if spec.dtype == tf.int32 else spec.dtype, name=spec.name) for (key, spec) in self.input_signature.items()}
                    int64_serving = self.serving.get_concrete_function(int64_spec)
                    signatures = {'serving_default': serving_default, 'int64_serving': int64_serving}
                else:
                    signatures = serving_default
            saved_model_dir = os.path.join(save_directory, 'saved_model', str(version))
            self.save(saved_model_dir, include_optimizer=False, signatures=signatures)
            logger.info(f'Saved model created in {saved_model_dir}')
        self.config.architectures = [self.__class__.__name__[2:]]
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self.config)
        self.config.save_pretrained(save_directory)
        if self.can_generate():
            self.generation_config.save_pretrained(save_directory)
        weights_name = SAFE_WEIGHTS_NAME if safe_serialization else TF2_WEIGHTS_NAME
        output_model_file = os.path.join(save_directory, weights_name)
        (shards, index) = tf_shard_checkpoint(self.weights, max_shard_size)
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            weights_no_suffix = weights_name.replace('.bin', '').replace('.safetensors', '')
            if filename.startswith(weights_no_suffix) and os.path.isfile(full_filename) and (filename not in shards.keys()):
                os.remove(full_filename)
        if index is None:
            if safe_serialization:
                state_dict = {strip_model_name_and_prefix(w.name): w.value() for w in self.weights}
                safe_save_file(state_dict, output_model_file, metadata={'format': 'tf'})
            else:
                self.save_weights(output_model_file)
            logger.info(f'Model weights saved in {output_model_file}')
        else:
            save_index_file = os.path.join(save_directory, TF2_WEIGHTS_INDEX_NAME)
            with open(save_index_file, 'w', encoding='utf-8') as index_file:
                content = json.dumps(index, indent=2, sort_keys=True) + '\n'
                index_file.write(content)
            logger.info(f'The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the index located at {save_index_file}.')
            for (shard_file, shard) in shards.items():
                with h5py.File(os.path.join(save_directory, shard_file), mode='w') as shard_file:
                    layers = []
                    for layer in sorted(shard, key=lambda x: x.name):
                        if 'model.' in layer.name or len(layer.name.split('/')) == 1:
                            layer_name = layer.name
                        else:
                            layer_name = '/'.join(layer.name.split('/')[1:])
                        param_dset = shard_file.create_dataset(layer_name, layer.numpy().shape, dtype=layer.numpy().dtype)
                        param_dset[:] = layer.numpy()
                        layers.append(layer_name.encode('utf8'))
                    save_attributes_to_hdf5_group(shard_file, 'layer_names', layers)
        if push_to_hub:
            self._upload_modified_files(save_directory, repo_id, files_timestamps, commit_message=commit_message, token=token)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, config: Optional[Union[PretrainedConfig, str, os.PathLike]]=None, cache_dir: Optional[Union[str, os.PathLike]]=None, ignore_mismatched_sizes: bool=False, force_download: bool=False, local_files_only: bool=False, token: Optional[Union[str, bool]]=None, revision: str='main', **kwargs):
        if False:
            return 10
        '\n        Instantiate a pretrained TF 2.0 model from a pre-trained model configuration.\n\n        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come\n        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning\n        task.\n\n        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those\n        weights are discarded.\n\n        Parameters:\n            pretrained_model_name_or_path (`str`, *optional*):\n                Can be either:\n\n                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.\n                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a\n                      user or organization name, like `dbmdz/bert-base-german-cased`.\n                    - A path to a *directory* containing model weights saved using\n                      [`~TFPreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.\n                    - A path or url to a *PyTorch state_dict save file* (e.g, `./pt_model/pytorch_model.bin`). In this\n                      case, `from_pt` should be set to `True` and a configuration object should be provided as `config`\n                      argument. This loading path is slower than converting the PyTorch model in a TensorFlow model\n                      using the provided conversion scripts and loading the TensorFlow model afterwards.\n                    - `None` if you are both providing the configuration and state dictionary (resp. with keyword\n                      arguments `config` and `state_dict`).\n            model_args (sequence of positional arguments, *optional*):\n                All remaining positional arguments will be passed to the underlying model\'s `__init__` method.\n            config (`Union[PretrainedConfig, str]`, *optional*):\n                Can be either:\n\n                    - an instance of a class derived from [`PretrainedConfig`],\n                    - a string valid as input to [`~PretrainedConfig.from_pretrained`].\n\n                Configuration for the model to use instead of an automatically loaded configuration. Configuration can\n                be automatically loaded when:\n\n                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained\n                      model).\n                    - The model was saved using [`~TFPreTrainedModel.save_pretrained`] and is reloaded by supplying the\n                      save directory.\n                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a\n                      configuration JSON file named *config.json* is found in the directory.\n            from_pt (`bool`, *optional*, defaults to `False`):\n                Load the model weights from a PyTorch state_dict save file (see docstring of\n                `pretrained_model_name_or_path` argument).\n            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):\n                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size\n                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a\n                checkpoint with 3 labels).\n            cache_dir (`str`, *optional*):\n                Path to a directory in which a downloaded pretrained model configuration should be cached if the\n                standard cache should not be used.\n            force_download (`bool`, *optional*, defaults to `False`):\n                Whether or not to force the (re-)download of the model weights and configuration files, overriding the\n                cached versions if they exist.\n            resume_download (`bool`, *optional*, defaults to `False`):\n                Whether or not to delete incompletely received files. Will attempt to resume the download if such a\n                file exists.\n            proxies:\n                (`Dict[str, str], `optional`): A dictionary of proxy servers to use by protocol or endpoint, e.g.,\n                `{\'http\': \'foo.bar:3128\', \'http://hostname\': \'foo.bar:4012\'}`. The proxies are used on each request.\n                output_loading_info(`bool`, *optional*, defaults to `False`): Whether ot not to also return a\n                dictionary containing missing keys, unexpected keys and error messages.\n            local_files_only(`bool`, *optional*, defaults to `False`):\n                Whether or not to only look at local files (e.g., not try downloading the model).\n            token (`str` or `bool`, *optional*):\n                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use\n                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).\n            revision (`str`, *optional*, defaults to `"main"`):\n                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a\n                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any\n                identifier allowed by git.\n\n\n                <Tip>\n\n                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".\n\n                </Tip>\n\n            mirror (`str`, *optional*):\n                Mirror source to accelerate downloads in China. If you are from China and have an accessibility\n                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.\n                Please refer to the mirror site for more information.\n            subfolder (`str`, *optional*, defaults to `""`):\n                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can\n                specify the folder name here.\n            tf_to_pt_weight_rename (`Callable`, *optional*):\n                A function that is called to transform the names of weights during the PyTorch to TensorFlow\n                crossloading process. This is not necessary for most models, but is useful to allow composite models to\n                be crossloaded correctly.\n            kwargs (remaining dictionary of keyword arguments, *optional*):\n                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,\n                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or\n                automatically loaded:\n\n                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the\n                      underlying model\'s `__init__` method (we assume all relevant updates to the configuration have\n                      already been done)\n                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class\n                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that\n                      corresponds to a configuration attribute will be used to override said attribute with the\n                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute\n                      will be passed to the underlying model\'s `__init__` function.\n\n        Examples:\n\n        ```python\n        >>> from transformers import BertConfig, TFBertModel\n\n        >>> # Download model and configuration from huggingface.co and cache.\n        >>> model = TFBertModel.from_pretrained("bert-base-uncased")\n        >>> # Model was saved using *save_pretrained(\'./test/saved_model/\')* (for example purposes, not runnable).\n        >>> model = TFBertModel.from_pretrained("./test/saved_model/")\n        >>> # Update configuration during loading.\n        >>> model = TFBertModel.from_pretrained("bert-base-uncased", output_attentions=True)\n        >>> assert model.config.output_attentions == True\n        >>> # Loading from a Pytorch model file instead of a TensorFlow checkpoint (slower, for example purposes, not runnable).\n        >>> config = BertConfig.from_json_file("./pt_model/my_pt_model_config.json")\n        >>> model = TFBertModel.from_pretrained("./pt_model/my_pytorch_model.bin", from_pt=True, config=config)\n        ```'
        from_pt = kwargs.pop('from_pt', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)
        output_loading_info = kwargs.pop('output_loading_info', False)
        use_auth_token = kwargs.pop('use_auth_token', None)
        trust_remote_code = kwargs.pop('trust_remote_code', None)
        _ = kwargs.pop('mirror', None)
        load_weight_prefix = kwargs.pop('load_weight_prefix', None)
        from_pipeline = kwargs.pop('_from_pipeline', None)
        from_auto_class = kwargs.pop('_from_auto', False)
        subfolder = kwargs.pop('subfolder', '')
        commit_hash = kwargs.pop('_commit_hash', None)
        tf_to_pt_weight_rename = kwargs.pop('tf_to_pt_weight_rename', None)
        _ = kwargs.pop('adapter_kwargs', None)
        if use_auth_token is not None:
            warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
            if token is not None:
                raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
            token = use_auth_token
        if trust_remote_code is True:
            logger.warning('The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.')
        user_agent = {'file_type': 'model', 'framework': 'tensorflow', 'from_auto_class': from_auto_class}
        if from_pipeline is not None:
            user_agent['using_pipeline'] = from_pipeline
        if is_offline_mode() and (not local_files_only):
            logger.info('Offline mode: forcing local_files_only=True')
            local_files_only = True
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            (config, model_kwargs) = cls.config_class.from_pretrained(config_path, cache_dir=cache_dir, return_unused_kwargs=True, force_download=force_download, resume_download=resume_download, proxies=proxies, local_files_only=local_files_only, token=token, revision=revision, _from_auto=from_auto_class, _from_pipeline=from_pipeline, _commit_hash=commit_hash, **kwargs)
        else:
            model_kwargs = kwargs
        if commit_hash is None:
            commit_hash = getattr(config, '_commit_hash', None)
        is_sharded = False
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)
            if is_local:
                if from_pt and os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                elif from_pt and os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)
                    is_sharded = True
                elif is_safetensors_available() and os.path.isfile(os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_INDEX_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_INDEX_NAME)
                    is_sharded = True
                elif is_safetensors_available() and os.path.isfile(os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_INDEX_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_INDEX_NAME)
                    is_sharded = True
                    raise NotImplementedError('Support for sharded checkpoints using safetensors is coming soon!')
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)) or os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)):
                    raise EnvironmentError(f'Error no file named {TF2_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} but there is a file for PyTorch weights. Use `from_pt=True` to load this model from those weights.')
                else:
                    raise EnvironmentError(f'Error no file named {TF2_WEIGHTS_NAME} or {WEIGHTS_NAME} found in directory {pretrained_model_name_or_path}.')
            elif os.path.isfile(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
                is_local = True
            elif os.path.isfile(pretrained_model_name_or_path + '.index'):
                archive_file = pretrained_model_name_or_path + '.index'
                is_local = True
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = download_url(pretrained_model_name_or_path)
            else:
                if from_pt:
                    filename = WEIGHTS_NAME
                elif is_safetensors_available():
                    filename = SAFE_WEIGHTS_NAME
                else:
                    filename = TF2_WEIGHTS_NAME
                try:
                    cached_file_kwargs = {'cache_dir': cache_dir, 'force_download': force_download, 'proxies': proxies, 'resume_download': resume_download, 'local_files_only': local_files_only, 'token': token, 'user_agent': user_agent, 'revision': revision, 'subfolder': subfolder, '_raise_exceptions_for_missing_entries': False, '_commit_hash': commit_hash}
                    resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)
                    if resolved_archive_file is None and filename == SAFE_WEIGHTS_NAME:
                        filename = TF2_WEIGHTS_NAME
                        resolved_archive_file = cached_file(pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **cached_file_kwargs)
                    if resolved_archive_file is None and filename == TF2_WEIGHTS_NAME:
                        resolved_archive_file = cached_file(pretrained_model_name_or_path, TF2_WEIGHTS_INDEX_NAME, **cached_file_kwargs)
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if resolved_archive_file is None and filename == WEIGHTS_NAME:
                        resolved_archive_file = cached_file(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME, **cached_file_kwargs)
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if resolved_archive_file is None:
                        has_file_kwargs = {'revision': revision, 'proxies': proxies, 'token': token}
                        if has_file(pretrained_model_name_or_path, SAFE_WEIGHTS_INDEX_NAME, **has_file_kwargs):
                            is_sharded = True
                            raise NotImplementedError('Support for sharded checkpoints using safetensors is coming soon!')
                        elif has_file(pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs):
                            raise EnvironmentError(f'{pretrained_model_name_or_path} does not appear to have a file named {TF2_WEIGHTS_NAME} but there is a file for PyTorch weights. Use `from_pt=True` to load this model from those weights.')
                        else:
                            raise EnvironmentError(f'{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME} or {TF_WEIGHTS_NAME}')
                except EnvironmentError:
                    raise
                except Exception:
                    raise EnvironmentError(f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME} or {TF_WEIGHTS_NAME}")
            if is_local:
                logger.info(f'loading weights file {archive_file}')
                resolved_archive_file = archive_file
                filename = resolved_archive_file.split(os.path.sep)[-1]
            else:
                logger.info(f'loading weights file {filename} from cache at {resolved_archive_file}')
        else:
            resolved_archive_file = None
        if is_sharded:
            (resolved_archive_file, _) = get_checkpoint_shard_files(pretrained_model_name_or_path, resolved_archive_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only, token=token, user_agent=user_agent, revision=revision, _commit_hash=commit_hash)
        safetensors_from_pt = False
        if filename == SAFE_WEIGHTS_NAME:
            with safe_open(resolved_archive_file, framework='tf') as f:
                safetensors_metadata = f.metadata()
            if safetensors_metadata is None or safetensors_metadata.get('format') not in ['pt', 'tf', 'flax']:
                raise OSError(f'The safetensors archive passed at {resolved_archive_file} does not contain the valid metadata. Make sure you save your model with the `save_pretrained` method.')
            safetensors_from_pt = safetensors_metadata.get('format') == 'pt'
        config.name_or_path = pretrained_model_name_or_path
        if cls._requires_load_weight_prefix and model_kwargs.get('name') is not None:
            model_kwargs['load_weight_prefix'] = load_weight_prefix + '/' + model_kwargs.get('name')
        model = cls(config, *model_args, **model_kwargs)
        if from_pt:
            from .modeling_tf_pytorch_utils import load_pytorch_checkpoint_in_tf2_model
            return load_pytorch_checkpoint_in_tf2_model(model, resolved_archive_file, allow_missing_keys=True, output_loading_info=output_loading_info, _prefix=load_weight_prefix, tf_to_pt_weight_rename=tf_to_pt_weight_rename)
        if load_weight_prefix is not None:
            with tf.compat.v1.variable_scope(load_weight_prefix):
                model.build()
        else:
            model.build()
        if safetensors_from_pt:
            from .modeling_tf_pytorch_utils import load_pytorch_state_dict_in_tf2_model
            with safe_open(resolved_archive_file, framework='tf') as safetensors_archive:
                return load_pytorch_state_dict_in_tf2_model(model, safetensors_archive, tf_inputs=False, allow_missing_keys=True, output_loading_info=output_loading_info, _prefix=load_weight_prefix, ignore_mismatched_sizes=ignore_mismatched_sizes, tf_to_pt_weight_rename=tf_to_pt_weight_rename)
        try:
            if is_sharded:
                for file in resolved_archive_file:
                    (os.path.isfile(file), f'Error retrieving files {file}')
                (missing_keys, unexpected_keys, mismatched_keys) = load_tf_sharded_weights(model, resolved_archive_file, ignore_mismatched_sizes=ignore_mismatched_sizes, _prefix=load_weight_prefix)
            else:
                (missing_keys, unexpected_keys, mismatched_keys) = load_tf_weights(model, resolved_archive_file, ignore_mismatched_sizes=ignore_mismatched_sizes, _prefix=load_weight_prefix)
        except OSError as e:
            try:
                with open(resolved_archive_file) as f:
                    if f.read().startswith('version'):
                        raise OSError('You seem to have cloned a repository without having git-lfs installed. Please install git-lfs and run `git lfs install` followed by `git lfs pull` in the folder you cloned.')
                    else:
                        raise ValueError from e
            except (UnicodeDecodeError, ValueError):
                raise OSError('Unable to load weights from h5 file. If you tried to load a TF 2.0 model from a PyTorch checkpoint, please set from_pt=True. ')
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]
        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]
        if len(unexpected_keys) > 0:
            logger.warning(f'Some layers from the model checkpoint at {pretrained_model_name_or_path} were not used when initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).\n- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).')
        else:
            logger.warning(f'All model checkpoint layers were used when initializing {model.__class__.__name__}.\n')
        if len(missing_keys) > 0:
            logger.warning(f'Some layers of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.')
        elif len(mismatched_keys) == 0:
            logger.warning(f'All the layers of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint was trained on, you can already use {model.__class__.__name__} for predictions without further training.')
        if len(mismatched_keys) > 0:
            mismatched_warning = '\n'.join([f'- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated' for (key, shape1, shape2) in mismatched_keys])
            logger.warning(f'Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} and are newly initialized because the shapes did not match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.')
        if model.can_generate():
            try:
                model.generation_config = GenerationConfig.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir, force_download=force_download, resume_download=resume_download, proxies=proxies, local_files_only=local_files_only, token=token, revision=revision, subfolder=subfolder, _from_auto=from_auto_class, _from_pipeline=from_pipeline, **kwargs)
            except OSError:
                logger.info('Generation config file not found, using a generation config created from the model config.')
                pass
        if output_loading_info:
            loading_info = {'missing_keys': missing_keys, 'unexpected_keys': unexpected_keys, 'mismatched_keys': mismatched_keys}
            return (model, loading_info)
        return model

    def push_to_hub(self, repo_id: str, use_temp_dir: Optional[bool]=None, commit_message: Optional[str]=None, private: Optional[bool]=None, max_shard_size: Optional[Union[int, str]]='10GB', token: Optional[Union[bool, str]]=None, use_auth_token: Optional[Union[bool, str]]=None, create_pr: bool=False, **base_model_card_args) -> str:
        if False:
            while True:
                i = 10
        '\n        Upload the model files to the 🤗 Model Hub while synchronizing a local clone of the repo in `repo_path_or_name`.\n\n        Parameters:\n            repo_id (`str`):\n                The name of the repository you want to push your model to. It should contain your organization name\n                when pushing to a given organization.\n            use_temp_dir (`bool`, *optional*):\n                Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.\n                Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.\n            commit_message (`str`, *optional*):\n                Message to commit while pushing. Will default to `"Upload model"`.\n            private (`bool`, *optional*):\n                Whether or not the repository created should be private.\n            token (`bool` or `str`, *optional*):\n                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated\n                when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`\n                is not specified.\n            max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):\n                Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard\n                will then be each of size lower than this size. If expressed as a string, needs to be digits followed\n                by a unit (like `"5MB"`).\n            create_pr (`bool`, *optional*, defaults to `False`):\n                Whether or not to create a PR with the uploaded files or directly commit.\n\n        Examples:\n\n        ```python\n        from transformers import TFAutoModel\n\n        model = TFAutoModel.from_pretrained("bert-base-cased")\n\n        # Push the model to your namespace with the name "my-finetuned-bert".\n        model.push_to_hub("my-finetuned-bert")\n\n        # Push the model to an organization with the name "my-finetuned-bert".\n        model.push_to_hub("huggingface/my-finetuned-bert")\n        ```\n        '
        if use_auth_token is not None:
            warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
            if token is not None:
                raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
            token = use_auth_token
        if 'repo_path_or_name' in base_model_card_args:
            warnings.warn('The `repo_path_or_name` argument is deprecated and will be removed in v5 of Transformers. Use `repo_id` instead.')
            repo_id = base_model_card_args.pop('repo_path_or_name')
        repo_url = base_model_card_args.pop('repo_url', None)
        organization = base_model_card_args.pop('organization', None)
        if os.path.isdir(repo_id):
            working_dir = repo_id
            repo_id = repo_id.split(os.path.sep)[-1]
        else:
            working_dir = repo_id.split('/')[-1]
        repo_id = self._create_repo(repo_id, private=private, token=token, repo_url=repo_url, organization=organization)
        if use_temp_dir is None:
            use_temp_dir = not os.path.isdir(working_dir)
        with working_or_temp_dir(working_dir=working_dir, use_temp_dir=use_temp_dir) as work_dir:
            files_timestamps = self._get_files_timestamps(work_dir)
            self.save_pretrained(work_dir, max_shard_size=max_shard_size)
            if hasattr(self, 'history') and hasattr(self, 'create_model_card'):
                base_model_card_args = {'output_dir': work_dir, 'model_name': Path(repo_id).name}
                base_model_card_args.update(base_model_card_args)
                self.create_model_card(**base_model_card_args)
            self._upload_modified_files(work_dir, repo_id, files_timestamps, commit_message=commit_message, token=token, create_pr=create_pr)

    @classmethod
    def register_for_auto_class(cls, auto_class='TFAutoModel'):
        if False:
            while True:
                i = 10
        '\n        Register this class with a given auto class. This should only be used for custom models as the ones in the\n        library are already mapped with an auto class.\n\n        <Tip warning={true}>\n\n        This API is experimental and may have some slight breaking changes in the next releases.\n\n        </Tip>\n\n        Args:\n            auto_class (`str` or `type`, *optional*, defaults to `"TFAutoModel"`):\n                The auto class to register this new model with.\n        '
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__
        import transformers.models.auto as auto_module
        if not hasattr(auto_module, auto_class):
            raise ValueError(f'{auto_class} is not a valid auto class.')
        cls._auto_class = auto_class

class TFConv1D(tf.keras.layers.Layer):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`):
            The number of output features.
        nx (`int`):
            The number of input features.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation to use to initialize the weights.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments passed along to the `__init__` of `tf.keras.layers.Layer`.
    """

    def __init__(self, nf, nx, initializer_range=0.02, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.nf = nf
        self.nx = nx
        self.initializer_range = initializer_range

    def build(self, input_shape):
        if False:
            print('Hello World!')
        self.weight = self.add_weight('weight', shape=[self.nx, self.nf], initializer=get_initializer(self.initializer_range))
        self.bias = self.add_weight('bias', shape=[1, self.nf], initializer=tf.zeros_initializer())

    def call(self, x):
        if False:
            i = 10
            return i + 15
        (bz, sl) = shape_list(x)[:2]
        x = tf.reshape(x, [-1, self.nx])
        x = tf.matmul(x, self.weight) + self.bias
        x = tf.reshape(x, [bz, sl, self.nf])
        return x

class TFSharedEmbeddings(tf.keras.layers.Layer):
    """
    Construct shared token embeddings.

    The weights of the embedding layer is usually shared with the weights of the linear decoder when doing language
    modeling.

    Args:
        vocab_size (`int`):
            The size of the vocabulary, e.g., the number of unique tokens.
        hidden_size (`int`):
            The size of the embedding vectors.
        initializer_range (`float`, *optional*):
            The standard deviation to use when initializing the weights. If no value is provided, it will default to
            \\\\(1/\\sqrt{hidden\\_size}\\\\).
        kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments passed along to the `__init__` of `tf.keras.layers.Layer`.
    """

    def __init__(self, vocab_size: int, hidden_size: int, initializer_range: Optional[float]=None, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = hidden_size ** (-0.5) if initializer_range is None else initializer_range
        warnings.warn('`TFSharedEmbeddings` is scheduled for deletion in v4.32, use `tf.keras.layers.Embedding` instead.', DeprecationWarning)

    def build(self, input_shape):
        if False:
            return 10
        '\n        Build shared token embedding layer Shared weights logic adapted from\n        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24\n        '
        self.weight = self.add_weight('weight', shape=[self.vocab_size, self.hidden_size], initializer=get_initializer(self.initializer_range))
        super().build(input_shape)

    def get_config(self):
        if False:
            print('Hello World!')
        config = {'vocab_size': self.vocab_size, 'hidden_size': self.hidden_size, 'initializer_range': self.initializer_range}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs: tf.Tensor, mode: str='embedding') -> tf.Tensor:
        if False:
            print('Hello World!')
        '\n        Get token embeddings of inputs or decode final hidden state.\n\n        Args:\n            inputs (`tf.Tensor`):\n                In embedding mode, should be an int64 tensor with shape `[batch_size, length]`.\n\n                In linear mode, should be a float tensor with shape `[batch_size, length, hidden_size]`.\n            mode (`str`, defaults to `"embedding"`):\n               A valid value is either `"embedding"` or `"linear"`, the first one indicates that the layer should be\n               used as an embedding layer, the second one that the layer should be used as a linear decoder.\n\n        Returns:\n            `tf.Tensor`: In embedding mode, the output is a float32 embedding tensor, with shape `[batch_size, length,\n            embedding_size]`.\n\n            In linear mode, the output is a float32 with shape `[batch_size, length, vocab_size]`.\n\n        Raises:\n            ValueError: if `mode` is not valid.\n\n        Shared weights logic is adapted from\n        [here](https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24).\n        '
        if mode == 'embedding':
            return self._embedding(inputs)
        elif mode == 'linear':
            return self._linear(inputs)
        else:
            raise ValueError(f'mode {mode} is not valid.')

    def _embedding(self, input_ids):
        if False:
            return 10
        'Applies embedding based on inputs tensor.'
        return tf.gather(self.weight, input_ids)

    def _linear(self, inputs):
        if False:
            return 10
        '\n        Computes logits by running inputs through a linear layer.\n\n        Args:\n            inputs: A float32 tensor with shape [..., hidden_size]\n\n        Returns:\n            float32 tensor with shape [..., vocab_size].\n        '
        first_dims = shape_list(inputs)[:-1]
        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.weight, transpose_b=True)
        return tf.reshape(logits, first_dims + [self.vocab_size])

class TFSequenceSummary(tf.keras.layers.Layer):
    """
    Compute a single vector summary of a sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model. Relevant arguments in the config class of the model are (refer to the actual
            config class of your model for the default values it uses):

            - **summary_type** (`str`) -- The method to use to make this summary. Accepted values are:

                - `"last"` -- Take the last token hidden state (like XLNet)
                - `"first"` -- Take the first token hidden state (like Bert)
                - `"mean"` -- Take the mean of all tokens hidden states
                - `"cls_index"` -- Supply a Tensor of classification token position (GPT/GPT-2)
                - `"attn"` -- Not implemented now, use multi-head attention

            - **summary_use_proj** (`bool`) -- Add a projection after the vector extraction.
            - **summary_proj_to_labels** (`bool`) -- If `True`, the projection outputs to `config.num_labels` classes
              (otherwise to `config.hidden_size`).
            - **summary_activation** (`Optional[str]`) -- Set to `"tanh"` to add a tanh activation to the output,
              another string or `None` will add no activation.
            - **summary_first_dropout** (`float`) -- Optional dropout probability before the projection and activation.
            - **summary_last_dropout** (`float`)-- Optional dropout probability after the projection and activation.

        initializer_range (`float`, defaults to 0.02): The standard deviation to use to initialize the weights.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments passed along to the `__init__` of `tf.keras.layers.Layer`.
    """

    def __init__(self, config: PretrainedConfig, initializer_range: float=0.02, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.summary_type = config.summary_type if hasattr(config, 'summary_use_proj') else 'last'
        if self.summary_type == 'attn':
            raise NotImplementedError
        self.has_summary = hasattr(config, 'summary_use_proj') and config.summary_use_proj
        if self.has_summary:
            if hasattr(config, 'summary_proj_to_labels') and config.summary_proj_to_labels and (config.num_labels > 0):
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = tf.keras.layers.Dense(num_classes, kernel_initializer=get_initializer(initializer_range), name='summary')
        self.has_activation = False
        activation_string = getattr(config, 'summary_activation', None)
        if activation_string is not None:
            self.has_activation = True
            self.activation = get_tf_activation(activation_string)
        self.has_first_dropout = hasattr(config, 'summary_first_dropout') and config.summary_first_dropout > 0
        if self.has_first_dropout:
            self.first_dropout = tf.keras.layers.Dropout(config.summary_first_dropout)
        self.has_last_dropout = hasattr(config, 'summary_last_dropout') and config.summary_last_dropout > 0
        if self.has_last_dropout:
            self.last_dropout = tf.keras.layers.Dropout(config.summary_last_dropout)

    def call(self, inputs, cls_index=None, training=False):
        if False:
            print('Hello World!')
        if not isinstance(inputs, (dict, tuple, list)):
            hidden_states = inputs
        elif isinstance(inputs, (tuple, list)):
            hidden_states = inputs[0]
            cls_index = inputs[1] if len(inputs) > 1 else None
            assert len(inputs) <= 2, 'Too many inputs.'
        else:
            hidden_states = inputs.get('hidden_states')
            cls_index = inputs.get('cls_index', None)
        if self.summary_type == 'last':
            output = hidden_states[:, -1]
        elif self.summary_type == 'first':
            output = hidden_states[:, 0]
        elif self.summary_type == 'mean':
            output = tf.reduce_mean(hidden_states, axis=1)
        elif self.summary_type == 'cls_index':
            hidden_shape = shape_list(hidden_states)
            if cls_index is None:
                cls_index = tf.fill(hidden_shape[:-2], hidden_shape[-2] - 1)
            cls_shape = shape_list(cls_index)
            if len(cls_shape) <= len(hidden_shape) - 2:
                cls_index = tf.expand_dims(cls_index, axis=-1)
            output = tf.gather(hidden_states, cls_index, batch_dims=len(hidden_shape) - 2)
            output = tf.squeeze(output, axis=len(hidden_shape) - 2)
        elif self.summary_type == 'attn':
            raise NotImplementedError
        if self.has_first_dropout:
            output = self.first_dropout(output, training=training)
        if self.has_summary:
            output = self.summary(output)
        if self.has_activation:
            output = self.activation(output)
        if self.has_last_dropout:
            output = self.last_dropout(output, training=training)
        return output

def get_initializer(initializer_range: float=0.02) -> tf.keras.initializers.TruncatedNormal:
    if False:
        i = 10
        return i + 15
    '\n    Creates a `tf.keras.initializers.TruncatedNormal` with the given range.\n\n    Args:\n        initializer_range (*float*, defaults to 0.02): Standard deviation of the initializer range.\n\n    Returns:\n        `tf.keras.initializers.TruncatedNormal`: The truncated normal initializer.\n    '
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)