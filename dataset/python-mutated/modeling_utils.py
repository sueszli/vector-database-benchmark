import collections
import functools
import gc
import importlib.metadata
import inspect
import json
import os
import re
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from packaging import version
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, Identity
from torch.utils.checkpoint import checkpoint
from .activations import get_activation
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import GenerationConfig, GenerationMixin
from .integrations import PeftAdapterMixin, deepspeed_config, is_deepspeed_zero3_enabled
from .pytorch_utils import Conv1D, apply_chunking_to_forward, find_pruneable_heads_and_indices, id_tensor_storage, prune_conv1d_layer, prune_layer, prune_linear_layer
from .utils import ADAPTER_SAFE_WEIGHTS_NAME, ADAPTER_WEIGHTS_NAME, CONFIG_NAME, DUMMY_INPUTS, FLAX_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME, ContextManagers, ModelOutput, PushToHubMixin, cached_file, copy_func, download_url, extract_commit_hash, has_file, is_accelerate_available, is_auto_awq_available, is_auto_gptq_available, is_bitsandbytes_available, is_flash_attn_2_available, is_offline_mode, is_optimum_available, is_peft_available, is_remote_url, is_safetensors_available, is_torch_tpu_available, logging, replace_return_docstrings, strtobool
from .utils.hub import convert_file_size_to_int, get_checkpoint_shard_files
from .utils.import_utils import ENV_VARS_TRUE_VALUES, is_sagemaker_mp_enabled, is_torch_fx_proxy, is_torchdynamo_compiling
from .utils.quantization_config import AwqConfig, BitsAndBytesConfig, GPTQConfig, QuantizationMethod
from .utils.versions import require_version_core
XLA_USE_BF16 = os.environ.get('XLA_USE_BF16', '0').upper()
XLA_DOWNCAST_BF16 = os.environ.get('XLA_DOWNCAST_BF16', '0').upper()
if is_accelerate_available():
    from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
    from accelerate.hooks import add_hook_to_module
    from accelerate.utils import check_tied_parameters_on_same_device, find_tied_parameters, get_balanced_memory, get_max_memory, load_offloaded_weights, offload_weight, save_offload_index, set_module_tensor_to_device
if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
logger = logging.get_logger(__name__)
_init_weights = True

def is_fsdp_enabled():
    if False:
        while True:
            i = 10
    return torch.distributed.is_available() and torch.distributed.is_initialized() and (strtobool(os.environ.get('ACCELERATE_USE_FSDP', 'False')) == 1) and (strtobool(os.environ.get('FSDP_CPU_RAM_EFFICIENT_LOADING', 'False')) == 1)

def is_fsdp_enabled_and_dist_rank_0():
    if False:
        print('Hello World!')
    return is_fsdp_enabled() and int(os.environ.get('LOCAL_RANK', -1)) == 0
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION
    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse('1.10')
else:
    IS_SAGEMAKER_MP_POST_1_10 = False
if is_peft_available():
    from .utils import find_adapter_config_file

@contextmanager
def no_init_weights(_enable=True):
    if False:
        print('Hello World!')
    '\n    Context manager to globally disable weight initialization to speed up loading large models.\n\n    TODO(Patrick): Delete safety argument `_enable=True` at next major version. .\n    '
    global _init_weights
    old_init_weights = _init_weights
    if _enable:
        _init_weights = False
    try:
        yield
    finally:
        _init_weights = old_init_weights

def get_parameter_device(parameter: Union[nn.Module, GenerationMixin, 'ModuleUtilsMixin']):
    if False:
        return 10
    try:
        return next(parameter.parameters()).device
    except StopIteration:

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            if False:
                for i in range(10):
                    print('nop')
            tuples = [(k, v) for (k, v) in module.__dict__.items() if torch.is_tensor(v)]
            return tuples
        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device

def get_first_parameter_dtype(parameter: Union[nn.Module, GenerationMixin, 'ModuleUtilsMixin']):
    if False:
        print('Hello World!')
    '\n    Returns the first parameter dtype (can be non-floating) or asserts if none were found.\n    '
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            if False:
                while True:
                    i = 10
            tuples = [(k, v) for (k, v) in module.__dict__.items() if torch.is_tensor(v)]
            return tuples
        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype

def get_parameter_dtype(parameter: Union[nn.Module, GenerationMixin, 'ModuleUtilsMixin']):
    if False:
        while True:
            i = 10
    '\n    Returns the first found floating dtype in parameters if there is one, otherwise returns the last dtype it found.\n    '
    last_dtype = None
    for t in parameter.parameters():
        last_dtype = t.dtype
        if t.is_floating_point():
            if XLA_USE_BF16 in ENV_VARS_TRUE_VALUES and is_torch_tpu_available():
                return torch.bfloat16
            if XLA_DOWNCAST_BF16 in ENV_VARS_TRUE_VALUES and is_torch_tpu_available():
                if t.dtype == torch.float:
                    return torch.bfloat16
                if t.dtype == torch.double:
                    return torch.float32
            return t.dtype
    if last_dtype is not None:
        return last_dtype

    def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
        if False:
            while True:
                i = 10
        tuples = [(k, v) for (k, v) in module.__dict__.items() if torch.is_tensor(v)]
        return tuples
    gen = parameter._named_members(get_members_fn=find_tensor_attributes)
    last_tuple = None
    for tuple in gen:
        last_tuple = tuple
        if tuple[1].is_floating_point():
            return tuple[1].dtype
    if last_tuple is not None:
        return last_tuple[1].dtype
    for t in parameter.buffers():
        last_dtype = t.dtype
        if t.is_floating_point():
            return t.dtype
    return last_dtype

def get_state_dict_float_dtype(state_dict):
    if False:
        while True:
            i = 10
    '\n    Returns the first found floating dtype in `state_dict` or asserts if none were found.\n    '
    for t in state_dict.values():
        if t.is_floating_point():
            return t.dtype
    raise ValueError("couldn't find any floating point dtypes in state_dict")

def get_state_dict_dtype(state_dict):
    if False:
        return 10
    '\n    Returns the first found floating dtype in `state_dict` if there is one, otherwise returns the first dtype.\n    '
    for t in state_dict.values():
        if t.is_floating_point():
            return t.dtype
    else:
        return next(state_dict.values()).dtype

def dtype_byte_size(dtype):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the size (in bytes) occupied by one parameter of type `dtype`.\n\n    Example:\n\n    ```py\n    >>> dtype_byte_size(torch.float32)\n    4\n    ```\n    '
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search('[^\\d](\\d+)$', str(dtype))
    if bit_search is None:
        raise ValueError(f'`dtype` is not a valid dtype: {dtype}.')
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8

def shard_checkpoint(state_dict: Dict[str, torch.Tensor], max_shard_size: Union[int, str]='10GB', weights_name: str=WEIGHTS_NAME):
    if False:
        return 10
    '\n    Splits a model state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a\n    given size.\n\n    The sub-checkpoints are determined by iterating through the `state_dict` in the order of its keys, so there is no\n    optimization made to make each sub-checkpoint as close as possible to the maximum size passed. For example, if the\n    limit is 10GB and we have weights of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as [6GB], [6+2GB],\n    [6+2+2GB] and not [6+2+2GB], [6+2GB], [6GB].\n\n    <Tip warning={true}>\n\n    If one of the model\'s weight is bigger than `max_shard_size`, it will end up in its own sub-checkpoint which will\n    have a size greater than `max_shard_size`.\n\n    </Tip>\n\n    Args:\n        state_dict (`Dict[str, torch.Tensor]`): The state dictionary of a model to save.\n        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):\n            The maximum size of each sub-checkpoint. If expressed as a string, needs to be digits followed by a unit\n            (like `"5MB"`).\n        weights_name (`str`, *optional*, defaults to `"pytorch_model.bin"`):\n            The name of the model save file.\n    '
    max_shard_size = convert_file_size_to_int(max_shard_size)
    sharded_state_dicts = [{}]
    last_block_size = 0
    total_size = 0
    storage_id_to_block = {}
    for (key, weight) in state_dict.items():
        if isinstance(weight, str):
            continue
        else:
            storage_id = id_tensor_storage(weight)
        if storage_id in storage_id_to_block:
            block_id = storage_id_to_block[storage_id]
            sharded_state_dicts[block_id][key] = weight
            continue
        weight_size = weight.numel() * dtype_byte_size(weight.dtype)
        if last_block_size + weight_size > max_shard_size and len(sharded_state_dicts[-1]) > 0:
            sharded_state_dicts.append({})
            last_block_size = 0
        sharded_state_dicts[-1][key] = weight
        last_block_size += weight_size
        total_size += weight_size
        storage_id_to_block[storage_id] = len(sharded_state_dicts) - 1
    if len(sharded_state_dicts) == 1:
        return ({weights_name: sharded_state_dicts[0]}, None)
    weight_map = {}
    shards = {}
    for (idx, shard) in enumerate(sharded_state_dicts):
        shard_file = weights_name.replace('.bin', f'-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.bin')
        shard_file = shard_file.replace('.safetensors', f'-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.safetensors')
        shards[shard_file] = shard
        for key in shard.keys():
            weight_map[key] = shard_file
    metadata = {'total_size': total_size}
    index = {'metadata': metadata, 'weight_map': weight_map}
    return (shards, index)

def load_sharded_checkpoint(model, folder, strict=True, prefer_safe=True):
    if False:
        return 10
    '\n    This is the same as\n    [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict)\n    but for a sharded checkpoint.\n\n    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being\n    loaded in the model.\n\n    Args:\n        model (`torch.nn.Module`): The model in which to load the checkpoint.\n        folder (`str` or `os.PathLike`): A path to a folder containing the sharded checkpoint.\n        strict (`bool`, *optional`, defaults to `True`):\n            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.\n        prefer_safe (`bool`, *optional*, defaults to `False`)\n            If both safetensors and PyTorch save files are present in checkpoint and `prefer_safe` is True, the\n            safetensors files will be loaded. Otherwise, PyTorch files are always loaded when possible.\n\n    Returns:\n        `NamedTuple`: A named tuple with `missing_keys` and `unexpected_keys` fields\n            - `missing_keys` is a list of str containing the missing keys\n            - `unexpected_keys` is a list of str containing the unexpected keys\n    '
    index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)
    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)
    if not index_present and (not (safe_index_present and is_safetensors_available())):
        filenames = (WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME) if is_safetensors_available() else (WEIGHTS_INDEX_NAME,)
        raise ValueError(f"Can't find a checkpoint index ({' or '.join(filenames)}) in {folder}.")
    load_safe = False
    if safe_index_present:
        if prefer_safe:
            if is_safetensors_available():
                load_safe = True
            else:
                logger.warning(f'Cannot load sharded checkpoint at {folder} safely since safetensors is not installed!')
        elif not index_present:
            load_safe = True
    load_index = safe_index_file if load_safe else index_file
    with open(load_index, 'r', encoding='utf-8') as f:
        index = json.load(f)
    shard_files = list(set(index['weight_map'].values()))
    loaded_keys = index['weight_map'].keys()
    model_keys = model.state_dict().keys()
    missing_keys = [key for key in model_keys if key not in loaded_keys]
    unexpected_keys = [key for key in loaded_keys if key not in model_keys]
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = f'Error(s) in loading state_dict for {model.__class__.__name__}'
        if len(missing_keys) > 0:
            str_missing_keys = ','.join([f'"{k}"' for k in missing_keys])
            error_message += f'\nMissing key(s): {str_missing_keys}.'
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ','.join([f'"{k}"' for k in unexpected_keys])
            error_message += f'\nMissing key(s): {str_unexpected_keys}.'
        raise RuntimeError(error_message)
    loader = safe_load_file if load_safe else partial(torch.load, map_location='cpu')
    for shard_file in shard_files:
        state_dict = loader(os.path.join(folder, shard_file))
        model.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()
    return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)

def load_state_dict(checkpoint_file: Union[str, os.PathLike]):
    if False:
        return 10
    '\n    Reads a PyTorch checkpoint file, returning properly formatted errors if they arise.\n    '
    if checkpoint_file.endswith('.safetensors') and is_safetensors_available():
        with safe_open(checkpoint_file, framework='pt') as f:
            metadata = f.metadata()
        if metadata.get('format') not in ['pt', 'tf', 'flax']:
            raise OSError(f'The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure you save your model with the `save_pretrained` method.')
        return safe_load_file(checkpoint_file)
    try:
        if (is_deepspeed_zero3_enabled() or is_fsdp_enabled()) and torch.distributed.is_initialized() and (torch.distributed.get_rank() > 0):
            map_location = 'meta'
        else:
            map_location = 'cpu'
        return torch.load(checkpoint_file, map_location=map_location)
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read(7) == 'version':
                    raise OSError('You seem to have cloned a repository without having git-lfs installed. Please install git-lfs and run `git lfs install` followed by `git lfs pull` in the folder you cloned.')
                else:
                    raise ValueError(f'Unable to locate the file {checkpoint_file} which is necessary to load this pretrained model. Make sure you have saved the model properly.') from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(f"Unable to load weights from pytorch checkpoint file for '{checkpoint_file}' at '{checkpoint_file}'. If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True.")

def set_initialized_submodules(model, state_dict_keys):
    if False:
        for i in range(10):
            print('nop')
    '\n    Sets the `_is_hf_initialized` flag in all submodules of a given model when all its weights are in the loaded state\n    dict.\n    '
    for (module_name, module) in model.named_modules():
        loaded_keys = [k.replace(f'{module_name}.', '') for k in state_dict_keys if k.startswith(f'{module_name}.')]
        if len(set(module.state_dict().keys()) - set(loaded_keys)) == 0:
            module._is_hf_initialized = True

def _load_state_dict_into_model(model_to_load, state_dict, start_prefix):
    if False:
        i = 10
        return i + 15
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for (old_key, new_key) in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    error_msgs = []

    def load(module: nn.Module, state_dict, prefix=''):
        if False:
            print('Hello World!')
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            if is_deepspeed_zero3_enabled():
                import deepspeed
                named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
                params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
                if len(params_to_gather) > 0:
                    with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)
        for (name, child) in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + '.')
    load(model_to_load, state_dict, prefix=start_prefix)
    del state_dict
    return error_msgs

def find_submodule_and_param_name(model, long_key, start_prefix):
    if False:
        i = 10
        return i + 15
    "\n    A helper util to find the last sub-module and the param/buffer name. If `start_prefix` is supplied it'll be removed\n    from the start of the key\n    "
    if len(start_prefix) > 0 and long_key.startswith(start_prefix):
        long_key = '.'.join(long_key.split('.')[1:])
    split_key = long_key.split('.')
    submodule = model
    while len(split_key) > 1:
        if hasattr(submodule, split_key[0]):
            submodule = getattr(submodule, split_key[0])
            del split_key[0]
        else:
            submodule = None
            break
    if submodule == model:
        submodule = None
    return (submodule, split_key[0])

def _move_model_to_meta(model, loaded_state_dict_keys, start_prefix):
    if False:
        for i in range(10):
            print('nop')
    '\n    Moves `loaded_state_dict_keys` in model to meta device which frees up the memory taken by those params.\n\n    `start_prefix` is used for models which insert their name into model keys, e.g. `bert` in\n    `bert.pooler.dense.weight`\n\n    '
    for k in loaded_state_dict_keys:
        (submodule, param_name) = find_submodule_and_param_name(model, k, start_prefix)
        if submodule is not None:
            new_val = getattr(submodule, param_name)
            if isinstance(new_val, torch.nn.Parameter):
                new_val = torch.nn.Parameter(new_val.to('meta'))
            else:
                new_val = new_val.to('meta')
            setattr(submodule, param_name, new_val)

def _load_state_dict_into_meta_model(model, state_dict, loaded_state_dict_keys, start_prefix, expected_keys, device_map=None, offload_folder=None, offload_index=None, state_dict_folder=None, state_dict_index=None, dtype=None, is_quantized=False, is_safetensors=False, keep_in_fp32_modules=None):
    if False:
        return 10
    '\n    This is somewhat similar to `_load_state_dict_into_model`, but deals with a model that has some or all of its\n    params on a `meta` device. It replaces the model params with the data from the `state_dict`, while moving the\n    params back to the normal device, but only for `loaded_state_dict_keys`.\n\n    `start_prefix` is used for models which insert their name into model keys, e.g. `bert` in\n    `bert.pooler.dense.weight`\n\n    '
    if is_quantized:
        from .integrations import set_module_quantized_tensor_to_device
    error_msgs = []
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for (old_key, new_key) in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
    for (param_name, param) in state_dict.items():
        if param_name not in loaded_state_dict_keys or param_name not in expected_keys:
            continue
        if param_name.startswith(start_prefix):
            param_name = param_name[len(start_prefix):]
        module_name = param_name
        set_module_kwargs = {}
        if dtype is not None and torch.is_floating_point(param):
            if keep_in_fp32_modules is not None and any((module_to_keep_in_fp32 in param_name.split('.') for module_to_keep_in_fp32 in keep_in_fp32_modules)) and (dtype == torch.float16):
                param = param.to(torch.float32)
                if 'dtype' in list(inspect.signature(set_module_tensor_to_device).parameters):
                    set_module_kwargs['dtype'] = torch.float32
            else:
                param = param.to(dtype)
        if dtype is None:
            old_param = model
            splits = param_name.split('.')
            for split in splits:
                old_param = getattr(old_param, split)
                if old_param is None:
                    break
            if old_param is not None:
                param = param.to(old_param.dtype)
        set_module_kwargs['value'] = param
        if device_map is None:
            param_device = 'cpu'
        else:
            while len(module_name) > 0 and module_name not in device_map:
                module_name = '.'.join(module_name.split('.')[:-1])
            if module_name == '' and '' not in device_map:
                raise ValueError(f"{param_name} doesn't have any device set.")
            param_device = device_map[module_name]
        if param_device == 'disk':
            if not is_safetensors:
                offload_index = offload_weight(param, param_name, offload_folder, offload_index)
        elif param_device == 'cpu' and state_dict_index is not None:
            state_dict_index = offload_weight(param, param_name, state_dict_folder, state_dict_index)
        elif not is_quantized:
            set_module_tensor_to_device(model, param_name, param_device, **set_module_kwargs)
        else:
            if param.dtype == torch.int8 and param_name.replace('weight', 'SCB') in state_dict.keys():
                fp16_statistics = state_dict[param_name.replace('weight', 'SCB')]
            else:
                fp16_statistics = None
            if 'SCB' not in param_name:
                set_module_quantized_tensor_to_device(model, param_name, param_device, value=param, fp16_statistics=fp16_statistics)
    return (error_msgs, offload_index, state_dict_index)

def _add_variant(weights_name: str, variant: Optional[str]=None) -> str:
    if False:
        i = 10
        return i + 15
    if variant is not None:
        splits = weights_name.split('.')
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = '.'.join(splits)
    return weights_name

class ModuleUtilsMixin:
    """
    A few utilities for `torch.nn.Modules`, to be used as a mixin.
    """

    @staticmethod
    def _hook_rss_memory_pre_forward(module, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        try:
            import psutil
        except ImportError:
            raise ImportError('You need to install psutil (pip install psutil) to use memory tracing.')
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_pre_forward = mem.rss
        return None

    @staticmethod
    def _hook_rss_memory_post_forward(module, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        try:
            import psutil
        except ImportError:
            raise ImportError('You need to install psutil (pip install psutil) to use memory tracing.')
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_post_forward = mem.rss
        mem_rss_diff = module.mem_rss_post_forward - module.mem_rss_pre_forward
        module.mem_rss_diff = mem_rss_diff + (module.mem_rss_diff if hasattr(module, 'mem_rss_diff') else 0)
        return None

    def add_memory_hooks(self):
        if False:
            print('Hello World!')
        '\n        Add a memory hook before and after each sub-module forward pass to record increase in memory consumption.\n\n        Increase in memory consumption is stored in a `mem_rss_diff` attribute for each module and can be reset to zero\n        with `model.reset_memory_hooks_state()`.\n        '
        for module in self.modules():
            module.register_forward_pre_hook(self._hook_rss_memory_pre_forward)
            module.register_forward_hook(self._hook_rss_memory_post_forward)
        self.reset_memory_hooks_state()

    def reset_memory_hooks_state(self):
        if False:
            i = 10
            return i + 15
        '\n        Reset the `mem_rss_diff` attribute of each module (see [`~modeling_utils.ModuleUtilsMixin.add_memory_hooks`]).\n        '
        for module in self.modules():
            module.mem_rss_diff = 0
            module.mem_rss_post_forward = 0
            module.mem_rss_pre_forward = 0

    @property
    def device(self) -> torch.device:
        if False:
            return 10
        '\n        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same\n        device).\n        '
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        if False:
            for i in range(10):
                print('nop')
        '\n        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).\n        '
        return get_parameter_dtype(self)

    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        if False:
            return 10
        '\n        Invert an attention mask (e.g., switches 0. and 1.).\n\n        Args:\n            encoder_attention_mask (`torch.Tensor`): An attention mask.\n\n        Returns:\n            `torch.Tensor`: The inverted attention mask.\n        '
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min
        return encoder_extended_attention_mask

    @staticmethod
    def create_extended_attention_mask_for_decoder(input_shape, attention_mask, device=None):
        if False:
            i = 10
            return i + 15
        if device is not None:
            warnings.warn('The `device` argument is deprecated and will be removed in v5 of Transformers.', FutureWarning)
        else:
            device = attention_mask.device
        (batch_size, seq_length) = input_shape
        seq_ids = torch.arange(seq_length, device=device)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        causal_mask = causal_mask.to(attention_mask.dtype)
        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = torch.cat([torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype), causal_mask], axis=-1)
        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        return extended_attention_mask

    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int], device: torch.device=None, dtype: torch.float=None) -> Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.\n\n        Arguments:\n            attention_mask (`torch.Tensor`):\n                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.\n            input_shape (`Tuple[int]`):\n                The shape of the input to the model.\n\n        Returns:\n            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.\n        '
        if dtype is None:
            dtype = self.dtype
        if not (attention_mask.dim() == 2 and self.config.is_decoder):
            if device is not None:
                warnings.warn('The `device` argument is deprecated and will be removed in v5 of Transformers.', FutureWarning)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if self.config.is_decoder:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(input_shape, attention_mask, device)
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f'Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})')
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def get_head_mask(self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool=False) -> Tensor:
        if False:
            while True:
                i = 10
        '\n        Prepare the head mask if needed.\n\n        Args:\n            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):\n                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).\n            num_hidden_layers (`int`):\n                The number of hidden layers in the model.\n            is_attention_chunked (`bool`, *optional*, defaults to `False`):\n                Whether or not the attentions scores are computed by chunks or not.\n\n        Returns:\n            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with\n            `[None]` for each layer.\n        '
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        if False:
            for i in range(10):
                print('nop')
        '-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]'
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert head_mask.dim() == 5, f'head_mask.dim != 5, instead {head_mask.dim()}'
        head_mask = head_mask.to(dtype=self.dtype)
        return head_mask

    def num_parameters(self, only_trainable: bool=False, exclude_embeddings: bool=False) -> int:
        if False:
            return 10
        '\n        Get number of (optionally, trainable or non-embeddings) parameters in the module.\n\n        Args:\n            only_trainable (`bool`, *optional*, defaults to `False`):\n                Whether or not to return only the number of trainable parameters\n\n            exclude_embeddings (`bool`, *optional*, defaults to `False`):\n                Whether or not to return only the number of non-embeddings parameters\n\n        Returns:\n            `int`: The number of parameters.\n        '
        if exclude_embeddings:
            embedding_param_names = [f'{name}.weight' for (name, module_type) in self.named_modules() if isinstance(module_type, nn.Embedding)]
            total_parameters = [parameter for (name, parameter) in self.named_parameters() if name not in embedding_param_names]
        else:
            total_parameters = list(self.parameters())
        total_numel = []
        is_loaded_in_4bit = getattr(self, 'is_loaded_in_4bit', False)
        if is_loaded_in_4bit:
            if is_bitsandbytes_available():
                import bitsandbytes as bnb
            else:
                raise ValueError('bitsandbytes is not installed but it seems that the model has been loaded in 4bit precision, something went wrong make sure to install bitsandbytes with `pip install bitsandbytes`.')
        for param in total_parameters:
            if param.requires_grad or not only_trainable:
                if is_loaded_in_4bit and isinstance(param, bnb.nn.Params4bit):
                    total_numel.append(param.numel() * 2)
                else:
                    total_numel.append(param.numel())
        return sum(total_numel)

    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Helper function to estimate the total number of tokens from the model inputs.\n\n        Args:\n            inputs (`dict`): The model inputs.\n\n        Returns:\n            `int`: The total number of tokens.\n        '
        if not hasattr(self, 'warnings_issued'):
            self.warnings_issued = {}
        if self.main_input_name in input_dict:
            return input_dict[self.main_input_name].numel()
        elif 'estimate_tokens' not in self.warnings_issued:
            logger.warning('Could not estimate the number of tokens of the input, floating-point operations will not be computed')
            self.warnings_issued['estimate_tokens'] = True
        return 0

    def floating_point_ops(self, input_dict: Dict[str, Union[torch.Tensor, Any]], exclude_embeddings: bool=True) -> int:
        if False:
            while True:
                i = 10
        '\n        Get number of (optionally, non-embeddings) floating-point operations for the forward and backward passes of a\n        batch with this transformer model. Default approximation neglects the quadratic dependency on the number of\n        tokens (valid if `12 * d_model << sequence_length`) as laid out in [this\n        paper](https://arxiv.org/pdf/2001.08361.pdf) section 2.1. Should be overridden for transformers with parameter\n        re-use e.g. Albert or Universal Transformers, or if doing long-range modeling with very high sequence lengths.\n\n        Args:\n            batch_size (`int`):\n                The batch size for the forward pass.\n\n            sequence_length (`int`):\n                The number of tokens in each line of the batch.\n\n            exclude_embeddings (`bool`, *optional*, defaults to `True`):\n                Whether or not to count embedding and softmax operations.\n\n        Returns:\n            `int`: The number of floating-point operations.\n        '
        return 6 * self.estimate_tokens(input_dict) * self.num_parameters(exclude_embeddings=exclude_embeddings)

class PreTrainedModel(nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin, PeftAdapterMixin):
    """
    Base class for all models.

    [`PreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models as well as a few methods common to all models to:

        - resize the input embeddings,
        - prune heads in the self-attention heads.

    Class attributes (overridden by derived classes):

        - **config_class** ([`PretrainedConfig`]) -- A subclass of [`PretrainedConfig`] to use as configuration class
          for this model architecture.
        - **load_tf_weights** (`Callable`) -- A python *method* for loading a TensorFlow checkpoint in a PyTorch model,
          taking as arguments:

            - **model** ([`PreTrainedModel`]) -- An instance of the model on which to load the TensorFlow checkpoint.
            - **config** ([`PreTrainedConfig`]) -- An instance of the configuration associated to the model.
            - **path** (`str`) -- A path to the TensorFlow checkpoint.

        - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
          classes of the same architecture adding modules on top of the base model.
        - **is_parallelizable** (`bool`) -- A flag indicating whether this model supports model parallelization.
        - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
          models, `pixel_values` for vision models and `input_values` for speech models).
    """
    config_class = None
    base_model_prefix = ''
    main_input_name = 'input_ids'
    _auto_class = None
    _no_split_modules = None
    _skip_keys_device_placement = None
    _keep_in_fp32_modules = None
    _keys_to_ignore_on_load_missing = None
    _keys_to_ignore_on_load_unexpected = None
    _keys_to_ignore_on_save = None
    _tied_weights_keys = None
    is_parallelizable = False
    supports_gradient_checkpointing = False
    _supports_flash_attn_2 = False

    @property
    def dummy_inputs(self) -> Dict[str, torch.Tensor]:
        if False:
            return 10
        '\n        `Dict[str, torch.Tensor]`: Dummy inputs to do a forward pass in the network.\n        '
        return {'input_ids': torch.tensor(DUMMY_INPUTS)}

    @property
    def framework(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        :str: Identifies that this is a PyTorch model.\n        '
        return 'pt'

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(f'Parameter config in `{self.__class__.__name__}(config)` should be an instance of class `PretrainedConfig`. To create a model from a pretrained model use `model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`')
        self.config = config
        self.name_or_path = config.name_or_path
        self.warnings_issued = {}
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None

    def post_init(self):
        if False:
            while True:
                i = 10
        "\n        A method executed at the end of each Transformer model initialization, to execute code that needs the model's\n        modules properly initialized (such as weight initialization).\n        "
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()

    def _backward_compatibility_gradient_checkpointing(self):
        if False:
            print('Hello World!')
        if self.supports_gradient_checkpointing and getattr(self.config, 'gradient_checkpointing', False):
            self.gradient_checkpointing_enable()
            delattr(self.config, 'gradient_checkpointing')

    @classmethod
    def _from_config(cls, config, **kwargs):
        if False:
            print('Hello World!')
        '\n        All context managers that the model should be initialized under go here.\n\n        Args:\n            torch_dtype (`torch.dtype`, *optional*):\n                Override the default `torch.dtype` and load the model under this dtype.\n        '
        torch_dtype = kwargs.pop('torch_dtype', None)
        dtype_orig = None
        if torch_dtype is not None:
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)
        if is_deepspeed_zero3_enabled():
            import deepspeed
            logger.info('Detected DeepSpeed ZeRO-3: activating zero.init() for this model')
            with deepspeed.zero.Init(config_dict_or_path=deepspeed_config()):
                model = cls(config, **kwargs)
        else:
            model = cls(config, **kwargs)
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)
        return model

    @classmethod
    def _set_default_torch_dtype(cls, dtype: torch.dtype) -> torch.dtype:
        if False:
            for i in range(10):
                print('nop')
        "\n        Change the default dtype and return the previous one. This is needed when wanting to instantiate the model\n        under specific dtype.\n\n        Args:\n            dtype (`torch.dtype`):\n                a floating dtype to set to.\n\n        Returns:\n            `torch.dtype`: the original `dtype` that can be used to restore `torch.set_default_dtype(dtype)` if it was\n            modified. If it wasn't, returns `None`.\n\n        Note `set_default_dtype` currently only works with floating-point types and asserts if for example,\n        `torch.int64` is passed. So if a non-float `dtype` is passed this functions will throw an exception.\n        "
        if not dtype.is_floating_point:
            raise ValueError(f"Can't instantiate {cls.__name__} model under dtype={dtype} since it is not a floating point dtype")
        logger.info(f'Instantiating {cls.__name__} model under default dtype {dtype}.')
        dtype_orig = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        return dtype_orig

    @property
    def base_model(self) -> nn.Module:
        if False:
            for i in range(10):
                print('nop')
        '\n        `torch.nn.Module`: The main body of the model.\n        '
        return getattr(self, self.base_model_prefix, self)

    @classmethod
    def can_generate(cls) -> bool:
        if False:
            while True:
                i = 10
        '\n        Returns whether this model can generate sequences with `.generate()`.\n\n        Returns:\n            `bool`: Whether this model can generate sequences with `.generate()`.\n        '
        if 'GenerationMixin' in str(cls.prepare_inputs_for_generation) and 'GenerationMixin' in str(cls.generate):
            return False
        return True

    @classmethod
    def _check_and_enable_flash_attn_2(cls, config, torch_dtype: Optional[torch.dtype]=None, device_map: Optional[Union[str, Dict[str, int]]]=None) -> PretrainedConfig:
        if False:
            return 10
        "\n        If you don't know about Flash Attention, check out the official repository of flash attention:\n        https://github.com/Dao-AILab/flash-attention\n\n        For using Flash Attention 1.0 you can do it directly via the `BetterTransformer` API, have a look at this\n        specific section of the documentation to learn more about it:\n        https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#decoder-models\n\n        The method checks if the current setup is compatible with Flash Attention as it requires the model to be in\n        half precision and not ran on CPU.\n\n        If all checks pass, the method will create an attribute in the config `_flash_attn_2_enabled` so that the model\n        can initialize the correct attention module\n        "
        if not cls._supports_flash_attn_2:
            raise ValueError('The current architecture does not support Flash Attention 2.0. Please open an issue on GitHub to request support for this architecture: https://github.com/huggingface/transformers/issues/new')
        if not is_flash_attn_2_available():
            raise ImportError('Flash Attention 2 is not available. Please refer to the documentation of https://github.com/Dao-AILab/flash-attention for installing it. Make sure to have at least the version 2.1.0')
        else:
            flash_attention_version = version.parse(importlib.metadata.version('flash_attn'))
            is_flash_greater_than_2 = flash_attention_version >= version.parse('2.1.0')
            if not is_flash_greater_than_2:
                raise ValueError(f'You need flash_attn package version to be greater or equal than 2.1. Make sure to have that version installed - detected version {flash_attention_version}')
        _is_bettertransformer = getattr(cls, 'use_bettertransformer', False)
        if _is_bettertransformer:
            raise ValueError('Flash Attention 2 and BetterTransformer API are not compatible. Please make sure to disable BetterTransformers by doing model.reverse_bettertransformer()')
        if torch_dtype is None:
            logger.warning('You are attempting to use Flash Attention 2.0 without specifying a torch dtype. This might lead to unexpected behaviour')
        elif torch_dtype is not None and torch_dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError(f'Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes. You passed {torch_dtype}, this might lead to unexpected behaviour.')
        if device_map is None:
            if torch.cuda.is_available():
                logger.warning("You are attempting to use Flash Attention 2.0 with a model initialized on CPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.")
            else:
                raise ValueError('You are attempting to use Flash Attention 2.0 with a model initialized on CPU and with no GPU available. This is not supported yet. Please make sure to have access to a GPU and either initialise the model on a GPU by passing a device_map or initialising the model on CPU and then moving it to GPU.')
        elif device_map is not None and isinstance(device_map, dict) and ('cpu' in device_map.values() or 'disk' in device_map.values()):
            raise ValueError('You are attempting to use Flash Attention 2.0 with a model dispatched on CPU or disk. This is not supported. Please make sure to initialise the model on a GPU by passing a device_map that contains only GPU devices as keys.')
        config._flash_attn_2_enabled = True
        return config

    def enable_input_require_grads(self):
        if False:
            i = 10
            return i + 15
        '\n        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping\n        the model weights fixed.\n        '

        def make_inputs_require_grads(module, input, output):
            if False:
                return 10
            output.requires_grad_(True)
        self._require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    def disable_input_require_grads(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Removes the `_require_grads_hook`.\n        '
        self._require_grads_hook.remove()

    def get_input_embeddings(self) -> nn.Module:
        if False:
            return 10
        "\n        Returns the model's input embeddings.\n\n        Returns:\n            `nn.Module`: A torch module mapping vocabulary to hidden states.\n        "
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError

    def set_input_embeddings(self, value: nn.Module):
        if False:
            while True:
                i = 10
        "\n        Set model's input embeddings.\n\n        Args:\n            value (`nn.Module`): A module mapping vocabulary to hidden states.\n        "
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            base_model.set_input_embeddings(value)
        else:
            raise NotImplementedError

    def get_output_embeddings(self) -> nn.Module:
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns the model's output embeddings.\n\n        Returns:\n            `nn.Module`: A torch module mapping hidden states to vocabulary.\n        "
        return None

    def _init_weights(self, module):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize the weights. This method should be overridden by derived class.\n        '
        pass

    def _initialize_weights(self, module):
        if False:
            i = 10
            return i + 15
        '\n        Initialize the weights if they are not already initialized.\n        '
        if getattr(module, '_is_hf_initialized', False):
            return
        self._init_weights(module)
        module._is_hf_initialized = True

    def tie_weights(self):
        if False:
            print('Hello World!')
        "\n        Tie the weights between the input embeddings and the output embeddings.\n\n        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning the\n        weights instead.\n        "
        if getattr(self.config, 'tie_word_embeddings', True):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())
        if getattr(self.config, 'is_encoder_decoder', False) and getattr(self.config, 'tie_encoder_decoder', False):
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)
        for module in self.modules():
            if hasattr(module, '_tie_weights'):
                module._tie_weights()

    @staticmethod
    def _tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str):
        if False:
            return 10
        uninitialized_encoder_weights: List[str] = []
        if decoder.__class__ != encoder.__class__:
            logger.info(f'{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized.')

        def tie_encoder_to_decoder_recursively(decoder_pointer: nn.Module, encoder_pointer: nn.Module, module_name: str, uninitialized_encoder_weights: List[str], depth=0):
            if False:
                for i in range(10):
                    print('nop')
            assert isinstance(decoder_pointer, nn.Module) and isinstance(encoder_pointer, nn.Module), f'{decoder_pointer} and {encoder_pointer} have to be of type nn.Module'
            if hasattr(decoder_pointer, 'weight'):
                assert hasattr(encoder_pointer, 'weight')
                encoder_pointer.weight = decoder_pointer.weight
                if hasattr(decoder_pointer, 'bias'):
                    assert hasattr(encoder_pointer, 'bias')
                    encoder_pointer.bias = decoder_pointer.bias
                return
            encoder_modules = encoder_pointer._modules
            decoder_modules = decoder_pointer._modules
            if len(decoder_modules) > 0:
                assert len(encoder_modules) > 0, f'Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}'
                all_encoder_weights = {module_name + '/' + sub_name for sub_name in encoder_modules.keys()}
                encoder_layer_pos = 0
                for (name, module) in decoder_modules.items():
                    if name.isdigit():
                        encoder_name = str(int(name) + encoder_layer_pos)
                        decoder_name = name
                        if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(encoder_modules) != len(decoder_modules):
                            encoder_layer_pos -= 1
                            continue
                    elif name not in encoder_modules:
                        continue
                    elif depth > 500:
                        raise ValueError('Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model.')
                    else:
                        decoder_name = encoder_name = name
                    tie_encoder_to_decoder_recursively(decoder_modules[decoder_name], encoder_modules[encoder_name], module_name + '/' + name, uninitialized_encoder_weights, depth=depth + 1)
                    all_encoder_weights.remove(module_name + '/' + encoder_name)
                uninitialized_encoder_weights += list(all_encoder_weights)
        tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights)
        if len(uninitialized_encoder_weights) > 0:
            logger.warning(f'The following encoder weights were not tied to the decoder {uninitialized_encoder_weights}')

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        if False:
            return 10
        'Tie or clone module weights depending of whether we are using TorchScript or not'
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight
        if getattr(output_embeddings, 'bias', None) is not None:
            output_embeddings.bias.data = nn.functional.pad(output_embeddings.bias.data, (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]), 'constant', 0)
        if hasattr(output_embeddings, 'out_features') and hasattr(input_embeddings, 'num_embeddings'):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def _get_no_split_modules(self, device_map: str):
        if False:
            print('Hello World!')
        '\n        Get the modules of the model that should not be spit when using device_map. We iterate through the modules to\n        get the underlying `_no_split_modules`.\n\n        Args:\n            device_map (`str`):\n                The device map value. Options are ["auto", "balanced", "balanced_low_0", "sequential"]\n\n        Returns:\n            `List[str]`: List of modules that should not be split\n        '
        _no_split_modules = set()
        modules_to_check = [self]
        while len(modules_to_check) > 0:
            module = modules_to_check.pop(-1)
            if module.__class__.__name__ not in _no_split_modules:
                if isinstance(module, PreTrainedModel):
                    if module._no_split_modules is None:
                        raise ValueError(f"{module.__class__.__name__} does not support `device_map='{device_map}'`. To implement support, the model class needs to implement the `_no_split_modules` attribute.")
                    else:
                        _no_split_modules = _no_split_modules | set(module._no_split_modules)
                modules_to_check += list(module.children())
        return list(_no_split_modules)

    def resize_token_embeddings(self, new_num_tokens: Optional[int]=None, pad_to_multiple_of: Optional[int]=None) -> nn.Embedding:
        if False:
            i = 10
            return i + 15
        '\n        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.\n\n        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.\n\n        Arguments:\n            new_num_tokens (`int`, *optional*):\n                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized\n                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just\n                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.\n            pad_to_multiple_of (`int`, *optional*):\n                If set will pad the embedding matrix to a multiple of the provided value.If `new_num_tokens` is set to\n                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.\n\n                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability\n                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more\n                details about this, or help on choosing the correct value for resizing, refer to this guide:\n                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc\n\n        Return:\n            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.\n        '
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds
        self.config.vocab_size = model_embeds.weight.shape[0]
        self.vocab_size = model_embeds.weight.shape[0]
        self.tie_weights()
        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        if False:
            return 10
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of)
        if hasattr(old_embeddings, '_hf_hook'):
            hook = old_embeddings._hf_hook
            add_hook_to_module(new_embeddings, hook)
        self.set_input_embeddings(new_embeddings)
        if pad_to_multiple_of is not None:
            if is_deepspeed_zero3_enabled():
                import deepspeed
                with deepspeed.zero.GatheredParameters(new_embeddings.weight, modifier_rank=None):
                    new_num_tokens = new_embeddings.weight.shape[0]
            else:
                new_num_tokens = new_embeddings.weight.shape[0]
        if self.get_output_embeddings() is not None and (not self.config.tie_word_embeddings):
            old_lm_head = self.get_output_embeddings()
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            if hasattr(old_lm_head, '_hf_hook'):
                hook = old_lm_head._hf_hook
                add_hook_to_module(new_lm_head, hook)
            self.set_output_embeddings(new_lm_head)
        return self.get_input_embeddings()

    def _get_resized_embeddings(self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int]=None, pad_to_multiple_of: Optional[int]=None) -> nn.Embedding:
        if False:
            return 10
        '\n        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly\n        initialized vectors at the end. Reducing the size will remove vectors from the end\n\n        Args:\n            old_embeddings (`torch.nn.Embedding`):\n                Old embeddings to be resized.\n            new_num_tokens (`int`, *optional*):\n                New number of tokens in the embedding matrix.\n\n                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove\n                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens\n                `torch.nn.Embedding` module of the model without doing anything.\n            pad_to_multiple_of (`int`, *optional*):\n                If set will pad the embedding matrix to a multiple of the provided value. If `new_num_tokens` is set to\n                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.\n\n                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability\n                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more\n                details about this, or help on choosing the correct value for resizing, refer to this guide:\n                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc\n\n\n        Return:\n            `torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if\n            `new_num_tokens` is `None`\n        '
        if pad_to_multiple_of is not None:
            if not isinstance(pad_to_multiple_of, int):
                raise ValueError(f'Asking to pad the embedding matrix to a multiple of `{pad_to_multiple_of}`, which is not and integer. Please make sure to pass an integer')
            if new_num_tokens is None:
                new_num_tokens = old_embeddings.weight.shape[0]
            new_num_tokens = (new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
        else:
            logger.info(f'You are resizing the embedding layer without providing a `pad_to_multiple_of` parameter. This means that the new embedding dimension will be {new_num_tokens}. This might induce some performance reduction as *Tensor Cores* will not be available. For more details about this, or help on choosing the correct value for resizing, refer to this guide: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc')
        if new_num_tokens is None:
            return old_embeddings
        if is_deepspeed_zero3_enabled():
            import deepspeed
            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=None):
                (old_num_tokens, old_embedding_dim) = old_embeddings.weight.size()
        else:
            (old_num_tokens, old_embedding_dim) = old_embeddings.weight.size()
        if old_num_tokens == new_num_tokens and (not is_deepspeed_zero3_enabled()):
            return old_embeddings
        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(f'Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You should either use a different resize function or make sure that `old_embeddings` are an instance of {nn.Embedding}.')
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim, device=old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
        self._init_weights(new_embeddings)
        n = min(old_num_tokens, new_num_tokens)
        if is_deepspeed_zero3_enabled():
            import deepspeed
            params = [old_embeddings.weight, new_embeddings.weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        else:
            new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        return new_embeddings

    def _get_resized_lm_head(self, old_lm_head: nn.Linear, new_num_tokens: Optional[int]=None, transposed: Optional[bool]=False) -> nn.Linear:
        if False:
            print('Hello World!')
        '\n        Build a resized Linear Module from a provided old Linear Module. Increasing the size will add newly initialized\n        vectors at the end. Reducing the size will remove vectors from the end\n\n        Args:\n            old_lm_head (`torch.nn.Linear`):\n                Old lm head liner layer to be resized.\n            new_num_tokens (`int`, *optional*):\n                New number of tokens in the linear matrix.\n\n                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove\n                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens\n                `torch.nn.Linear` module of the model without doing anything. transposed (`bool`, *optional*, defaults\n                to `False`): Whether `old_lm_head` is transposed or not. If True `old_lm_head.size()` is `lm_head_dim,\n                vocab_size` else `vocab_size, lm_head_dim`.\n\n        Return:\n            `torch.nn.Linear`: Pointer to the resized Linear Module or the old Linear Module if `new_num_tokens` is\n            `None`\n        '
        if new_num_tokens is None:
            return old_lm_head
        if is_deepspeed_zero3_enabled():
            import deepspeed
            with deepspeed.zero.GatheredParameters(old_lm_head.weight, modifier_rank=None):
                (old_num_tokens, old_lm_head_dim) = old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
        else:
            (old_num_tokens, old_lm_head_dim) = old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
        if old_num_tokens == new_num_tokens and (not is_deepspeed_zero3_enabled()):
            return old_lm_head
        if not isinstance(old_lm_head, nn.Linear):
            raise TypeError(f'Old language model head is of type {type(old_lm_head)}, which is not an instance of {nn.Linear}. You should either use a different resize function or make sure that `old_lm_head` are an instance of {nn.Linear}.')
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
        has_new_lm_head_bias = old_lm_head.bias is not None
        new_lm_head = nn.Linear(*new_lm_head_shape, bias=has_new_lm_head_bias, device=old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)
        self._init_weights(new_lm_head)
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        if is_deepspeed_zero3_enabled():
            import deepspeed
            params = [old_lm_head.weight, old_lm_head.bias, new_lm_head.weight, new_lm_head.bias]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                self._copy_lm_head_original_to_resized(new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias)
        else:
            self._copy_lm_head_original_to_resized(new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias)
        return new_lm_head

    def _copy_lm_head_original_to_resized(self, new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias):
        if False:
            for i in range(10):
                print('nop')
        if not transposed:
            new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        else:
            new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]
        if has_new_lm_head_bias:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        if False:
            return 10
        raise NotImplementedError(f'`resize_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`')

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        if False:
            print('Hello World!')
        raise NotImplementedError(f'`get_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`')

    def init_weights(self):
        if False:
            while True:
                i = 10
        '\n        If needed prunes and maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any\n        initialization logic in `_init_weights`.\n        '
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)
        if _init_weights:
            self.apply(self._initialize_weights)
            self.tie_weights()

    def prune_heads(self, heads_to_prune: Dict[int, List[int]]):
        if False:
            return 10
        '\n        Prunes heads of the base model.\n\n        Arguments:\n            heads_to_prune (`Dict[int, List[int]]`):\n                Dictionary with keys being selected layer indices (`int`) and associated values being the list of heads\n                to prune in said layer (list of `int`). For instance {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on\n                layer 1 and heads 2 and 3 on layer 2.\n        '
        for (layer, heads) in heads_to_prune.items():
            union_heads = set(self.config.pruned_heads.get(layer, [])) | set(heads)
            self.config.pruned_heads[layer] = list(union_heads)
        self.base_model._prune_heads(heads_to_prune)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if False:
            return 10
        '\n        Activates gradient checkpointing for the current model.\n\n        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint\n        activations".\n\n        We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of\n        the module. https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2\n\n        Args:\n            gradient_checkpointing_kwargs (dict, *optional*):\n                Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.\n        '
        if not self.supports_gradient_checkpointing:
            raise ValueError(f'{self.__class__.__name__} does not support gradient checkpointing.')
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
        if getattr(self, '_hf_peft_config_loaded', False):
            self.enable_input_require_grads()

    def _set_gradient_checkpointing(self, enable: bool=True, gradient_checkpointing_func: Callable=checkpoint):
        if False:
            return 10
        is_gradient_checkpointing_set = False
        if hasattr(self, 'gradient_checkpointing'):
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True
        for module in self.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True
        if not is_gradient_checkpointing_set:
            raise ValueError(f'{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute `gradient_checkpointing` to modules of the model that uses checkpointing.')

    def gradient_checkpointing_disable(self):
        if False:
            while True:
                i = 10
        '\n        Deactivates gradient checkpointing for the current model.\n\n        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint\n        activations".\n        '
        if self.supports_gradient_checkpointing:
            self._set_gradient_checkpointing(enable=False)
        if getattr(self, '_hf_peft_config_loaded', False):
            self.disable_input_require_grads()

    @property
    def is_gradient_checkpointing(self) -> bool:
        if False:
            return 10
        '\n        Whether gradient checkpointing is activated for this model or not.\n\n        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint\n        activations".\n        '
        return any((hasattr(m, 'gradient_checkpointing') and m.gradient_checkpointing for m in self.modules()))

    def save_pretrained(self, save_directory: Union[str, os.PathLike], is_main_process: bool=True, state_dict: Optional[dict]=None, save_function: Callable=torch.save, push_to_hub: bool=False, max_shard_size: Union[int, str]='5GB', safe_serialization: bool=True, variant: Optional[str]=None, token: Optional[Union[str, bool]]=None, save_peft_format: bool=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Save a model and its configuration file to a directory, so that it can be re-loaded using the\n        [`~PreTrainedModel.from_pretrained`] class method.\n\n        Arguments:\n            save_directory (`str` or `os.PathLike`):\n                Directory to which to save. Will be created if it doesn\'t exist.\n            is_main_process (`bool`, *optional*, defaults to `True`):\n                Whether the process calling this is the main process or not. Useful when in distributed training like\n                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on\n                the main process to avoid race conditions.\n            state_dict (nested dictionary of `torch.Tensor`):\n                The state dictionary of the model to save. Will default to `self.state_dict()`, but can be used to only\n                save parts of the model or if special precautions need to be taken when recovering the state dictionary\n                of a model (like when using model parallelism).\n            save_function (`Callable`):\n                The function to use to save the state dictionary. Useful on distributed training like TPUs when one\n                need to replace `torch.save` by another method.\n            push_to_hub (`bool`, *optional*, defaults to `False`):\n                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the\n                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your\n                namespace).\n            max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):\n                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size\n                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).\n                We default it to 5GB in order for models to be able to run easily on free-tier google colab instances\n                without CPU OOM issues.\n\n                <Tip warning={true}>\n\n                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard\n                which will be bigger than `max_shard_size`.\n\n                </Tip>\n\n            safe_serialization (`bool`, *optional*, defaults to `True`):\n                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).\n            variant (`str`, *optional*):\n                If specified, weights are saved in the format pytorch_model.<variant>.bin.\n            token (`str` or `bool`, *optional*):\n                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use\n                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).\n            save_peft_format (`bool`, *optional*, defaults to `True`):\n                For backward compatibility with PEFT library, in case adapter weights are attached to the model, all\n                keys of the state dict of adapters needs to be pre-pended with `base_model.model`. Advanced users can\n                disable this behaviours by setting `save_peft_format` to `False`.\n            kwargs (`Dict[str, Any]`, *optional*):\n                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.\n        '
        use_auth_token = kwargs.pop('use_auth_token', None)
        if use_auth_token is not None:
            warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
            if token is not None:
                raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
            token = use_auth_token
        if token is not None:
            kwargs['token'] = token
        _hf_peft_config_loaded = getattr(self, '_hf_peft_config_loaded', False)
        if getattr(self, 'is_loaded_in_8bit', False) and (not getattr(self, 'is_8bit_serializable', False)) and (not _hf_peft_config_loaded):
            raise ValueError('You are calling `save_pretrained` to a 8-bit converted model you may likely encounter unexepected behaviors. If you want to save 8-bit models, make sure to have `bitsandbytes>0.37.2` installed.')
        if getattr(self, 'is_loaded_in_4bit', False) and (not _hf_peft_config_loaded):
            raise NotImplementedError('You are calling `save_pretrained` on a 4-bit converted model. This is currently not supported')
        if 'save_config' in kwargs:
            warnings.warn('`save_config` is deprecated and will be removed in v5 of Transformers. Use `is_main_process` instead.')
            is_main_process = kwargs.pop('save_config')
        if safe_serialization and (not is_safetensors_available()):
            raise ImportError('`safe_serialization` requires the `safetensors library: `pip install safetensors`.')
        if os.path.isfile(save_directory):
            logger.error(f'Provided path ({save_directory}) should be a directory, not a file')
            return
        os.makedirs(save_directory, exist_ok=True)
        if push_to_hub:
            commit_message = kwargs.pop('commit_message', None)
            repo_id = kwargs.pop('repo_id', save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)
        model_to_save = unwrap_model(self)
        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.torch_dtype = str(dtype).split('.')[1]
        model_to_save.config.architectures = [model_to_save.__class__.__name__]
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self.config)
        if is_main_process:
            if not _hf_peft_config_loaded:
                model_to_save.config.save_pretrained(save_directory)
            if self.can_generate():
                model_to_save.generation_config.save_pretrained(save_directory)
            if _hf_peft_config_loaded:
                logger.info('Detected adapters on the model, saving the model in the PEFT format, only adapter weights will be saved.')
                state_dict = model_to_save.get_adapter_state_dict()
                if save_peft_format:
                    logger.info('To match the expected format of the PEFT library, all keys of the state dict of adapters will be pre-pended with `base_model.model`.')
                    peft_state_dict = {}
                    for (key, value) in state_dict.items():
                        peft_state_dict[f'base_model.model.{key}'] = value
                    state_dict = peft_state_dict
                active_adapter = self.active_adapters()
                if len(active_adapter) > 1:
                    raise ValueError('Multiple active adapters detected, saving multiple active adapters is not supported yet. You can save adapters separately one by one by iteratively calling `model.set_adapter(adapter_name)` then `model.save_pretrained(...)`')
                active_adapter = active_adapter[0]
                current_peft_config = self.peft_config[active_adapter]
                current_peft_config.save_pretrained(save_directory)
        if state_dict is None:
            state_dict = model_to_save.state_dict()
        if IS_SAGEMAKER_MP_POST_1_10:
            for (smp_to_hf, _) in smp.state.module_manager.translate_functions:
                state_dict = smp_to_hf(state_dict)
        if self._keys_to_ignore_on_save is not None:
            for ignore_key in self._keys_to_ignore_on_save:
                if ignore_key in state_dict.keys():
                    del state_dict[ignore_key]
        if safe_serialization:
            ptrs = collections.defaultdict(list)
            for (name, tensor) in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    ptrs[id_tensor_storage(tensor)].append(name)
                else:
                    ptrs[id(tensor)].append(name)
            shared_ptrs = {ptr: names for (ptr, names) in ptrs.items() if len(names) > 1}
            warn_names = set()
            for names in shared_ptrs.values():
                if self._tied_weights_keys is not None:
                    found = 0
                    for name in sorted(names):
                        matches_pattern = any((re.search(pat, name) for pat in self._tied_weights_keys))
                        if matches_pattern and name in state_dict:
                            found += 1
                            if found < len(names):
                                del state_dict[name]
                found = 0
                for name in names:
                    if name in state_dict:
                        found += 1
                        if found > 1:
                            del state_dict[name]
                            warn_names.add(name)
            if len(warn_names) > 0:
                logger.warning_once(f"Removed shared tensor {warn_names} while saving. This should be OK, but check by verifying that you don't receive any warning while reloading")
        if not _hf_peft_config_loaded:
            weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
            weights_name = _add_variant(weights_name, variant)
        else:
            weights_name = ADAPTER_SAFE_WEIGHTS_NAME if safe_serialization else ADAPTER_WEIGHTS_NAME
        (shards, index) = shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name=weights_name)
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            weights_no_suffix = weights_name.replace('.bin', '').replace('.safetensors', '')
            filename_no_suffix = filename.replace('.bin', '').replace('.safetensors', '')
            reg = re.compile('(.*?)-\\d{5}-of-\\d{5}')
            if filename.startswith(weights_no_suffix) and os.path.isfile(full_filename) and (filename not in shards.keys()) and is_main_process and (reg.fullmatch(filename_no_suffix) is not None):
                os.remove(full_filename)
        for (shard_file, shard) in shards.items():
            if safe_serialization:
                safe_save_file(shard, os.path.join(save_directory, shard_file), metadata={'format': 'pt'})
            else:
                save_function(shard, os.path.join(save_directory, shard_file))
        if index is None:
            path_to_weights = os.path.join(save_directory, _add_variant(WEIGHTS_NAME, variant))
            logger.info(f'Model weights saved in {path_to_weights}')
        else:
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, _add_variant(save_index_file, variant))
            with open(save_index_file, 'w', encoding='utf-8') as f:
                content = json.dumps(index, indent=2, sort_keys=True) + '\n'
                f.write(content)
            logger.info(f'The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the index located at {save_index_file}.')
        if push_to_hub:
            self._upload_modified_files(save_directory, repo_id, files_timestamps, commit_message=commit_message, token=token)

    def get_memory_footprint(self, return_buffers=True):
        if False:
            while True:
                i = 10
        '\n        Get the memory footprint of a model. This will return the memory footprint of the current model in bytes.\n        Useful to benchmark the memory footprint of the current model and design some tests. Solution inspired from the\n        PyTorch discussions: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2\n\n        Arguments:\n            return_buffers (`bool`, *optional*, defaults to `True`):\n                Whether to return the size of the buffer tensors in the computation of the memory footprint. Buffers\n                are tensors that do not require gradients and not registered as parameters. E.g. mean and std in batch\n                norm layers. Please see: https://discuss.pytorch.org/t/what-pytorch-means-by-buffers/120266/2\n        '
        mem = sum([param.nelement() * param.element_size() for param in self.parameters()])
        if return_buffers:
            mem_bufs = sum([buf.nelement() * buf.element_size() for buf in self.buffers()])
            mem = mem + mem_bufs
        return mem

    @wraps(torch.nn.Module.cuda)
    def cuda(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if getattr(self, 'quantization_method', None) == QuantizationMethod.BITS_AND_BYTES:
            raise ValueError('Calling `cuda()` is not supported for `4-bit` or `8-bit` quantized models. Please use the model as it is, since the model has already been set to the correct devices and casted to the correct `dtype`.')
        else:
            return super().cuda(*args, **kwargs)

    @wraps(torch.nn.Module.to)
    def to(self, *args, **kwargs):
        if False:
            return 10
        if getattr(self, 'quantization_method', None) == QuantizationMethod.BITS_AND_BYTES:
            raise ValueError('`.to` is not supported for `4-bit` or `8-bit` bitsandbytes models. Please use the model as it is, since the model has already been set to the correct devices and casted to the correct `dtype`.')
        elif getattr(self, 'quantization_method', None) == QuantizationMethod.GPTQ:
            dtype_present_in_args = False
            if 'dtype' not in kwargs:
                for arg in args:
                    if isinstance(arg, torch.dtype):
                        dtype_present_in_args = True
                        break
            else:
                dtype_present_in_args = True
            if dtype_present_in_args:
                raise ValueError('You cannot cast a GPTQ model in a new `dtype`. Make sure to load the model using `from_pretrained` using the desired `dtype` by passing the correct `torch_dtype` argument.')
        return super().to(*args, **kwargs)

    def half(self, *args):
        if False:
            i = 10
            return i + 15
        if getattr(self, 'is_quantized', False):
            raise ValueError('`.half()` is not supported for quantized model. Please use the model as it is, since the model has already been casted to the correct `dtype`.')
        else:
            return super().half(*args)

    def float(self, *args):
        if False:
            return 10
        if getattr(self, 'is_quantized', False):
            raise ValueError('`.float()` is not supported for quantized model. Please use the model as it is, since the model has already been casted to the correct `dtype`.')
        else:
            return super().float(*args)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, config: Optional[Union[PretrainedConfig, str, os.PathLike]]=None, cache_dir: Optional[Union[str, os.PathLike]]=None, ignore_mismatched_sizes: bool=False, force_download: bool=False, local_files_only: bool=False, token: Optional[Union[str, bool]]=None, revision: str='main', use_safetensors: bool=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Instantiate a pretrained pytorch model from a pre-trained model configuration.\n\n        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train\n        the model, you should first set it back in training mode with `model.train()`.\n\n        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come\n        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning\n        task.\n\n        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those\n        weights are discarded.\n\n        Parameters:\n            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):\n                Can be either:\n\n                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.\n                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a\n                      user or organization name, like `dbmdz/bert-base-german-cased`.\n                    - A path to a *directory* containing model weights saved using\n                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.\n                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In\n                      this case, `from_tf` should be set to `True` and a configuration object should be provided as\n                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a\n                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.\n                    - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,\n                      `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to\n                      `True`.\n                    - `None` if you are both providing the configuration and state dictionary (resp. with keyword\n                      arguments `config` and `state_dict`).\n            model_args (sequence of positional arguments, *optional*):\n                All remaining positional arguments will be passed to the underlying model\'s `__init__` method.\n            config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):\n                Can be either:\n\n                    - an instance of a class derived from [`PretrainedConfig`],\n                    - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].\n\n                Configuration for the model to use instead of an automatically loaded configuration. Configuration can\n                be automatically loaded when:\n\n                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained\n                      model).\n                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the\n                      save directory.\n                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a\n                      configuration JSON file named *config.json* is found in the directory.\n            state_dict (`Dict[str, torch.Tensor]`, *optional*):\n                A state dictionary to use instead of a state dictionary loaded from saved weights file.\n\n                This option can be used if you want to create a model from a pretrained configuration but load your own\n                weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and\n                [`~PreTrainedModel.from_pretrained`] is not a simpler option.\n            cache_dir (`Union[str, os.PathLike]`, *optional*):\n                Path to a directory in which a downloaded pretrained model configuration should be cached if the\n                standard cache should not be used.\n            from_tf (`bool`, *optional*, defaults to `False`):\n                Load the model weights from a TensorFlow checkpoint save file (see docstring of\n                `pretrained_model_name_or_path` argument).\n            from_flax (`bool`, *optional*, defaults to `False`):\n                Load the model weights from a Flax checkpoint save file (see docstring of\n                `pretrained_model_name_or_path` argument).\n            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):\n                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size\n                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a\n                checkpoint with 3 labels).\n            force_download (`bool`, *optional*, defaults to `False`):\n                Whether or not to force the (re-)download of the model weights and configuration files, overriding the\n                cached versions if they exist.\n            resume_download (`bool`, *optional*, defaults to `False`):\n                Whether or not to delete incompletely received files. Will attempt to resume the download if such a\n                file exists.\n            proxies (`Dict[str, str]`, *optional*):\n                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{\'http\': \'foo.bar:3128\',\n                \'http://hostname\': \'foo.bar:4012\'}`. The proxies are used on each request.\n            output_loading_info(`bool`, *optional*, defaults to `False`):\n                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.\n            local_files_only(`bool`, *optional*, defaults to `False`):\n                Whether or not to only look at local files (i.e., do not try to download the model).\n            token (`str` or `bool`, *optional*):\n                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use\n                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).\n            revision (`str`, *optional*, defaults to `"main"`):\n                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a\n                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any\n                identifier allowed by git.\n\n                <Tip>\n\n                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".\n\n                </Tip>\n\n            mirror (`str`, *optional*):\n                Mirror source to accelerate downloads in China. If you are from China and have an accessibility\n                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.\n                Please refer to the mirror site for more information.\n            _fast_init(`bool`, *optional*, defaults to `True`):\n                Whether or not to disable fast initialization.\n\n                <Tip warning={true}>\n\n                One should only disable *_fast_init* to ensure backwards compatibility with `transformers.__version__ <\n                4.6.0` for seeded model initialization. This argument will be removed at the next major version. See\n                [pull request 11471](https://github.com/huggingface/transformers/pull/11471) for more information.\n\n                </Tip>\n\n            > Parameters for big model inference\n\n            low_cpu_mem_usage(`bool`, *optional*):\n                Tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.\n                This is an experimental feature and a subject to change at any moment.\n            torch_dtype (`str` or `torch.dtype`, *optional*):\n                Override the default `torch.dtype` and load the model under a specific `dtype`. The different options\n                are:\n\n                1. `torch.float16` or `torch.bfloat16` or `torch.float`: load in a specified\n                  `dtype`, ignoring the model\'s `config.torch_dtype` if one exists. If not specified\n                  - the model will get loaded in `torch.float` (fp32).\n\n                2. `"auto"` - A `torch_dtype` entry in the `config.json` file of the model will be\n                  attempted to be used. If this entry isn\'t found then next check the `dtype` of the first weight in\n                  the checkpoint that\'s of a floating point type and use that as `dtype`. This will load the model\n                  using the `dtype` it was saved in at the end of the training. It can\'t be used as an indicator of how\n                  the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.\n\n                <Tip>\n\n                For some models the `dtype` they were trained in is unknown - you may try to check the model\'s paper or\n                reach out to the authors and ask them to add this information to the model\'s card and to insert the\n                `torch_dtype` entry in `config.json` on the hub.\n\n                </Tip>\n\n            device_map (`str` or `Dict[str, Union[int, str, torch.device]]` or `int` or `torch.device`, *optional*):\n                A map that specifies where each submodule should go. It doesn\'t need to be refined to each\n                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the\n                same device. If we only pass the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank\n                like `1`) on which the model will be allocated, the device map will map the entire model to this\n                device. Passing `device_map = 0` means put the whole model on GPU 0.\n\n                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For\n                more information about each option see [designing a device\n                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).\n            max_memory (`Dict`, *optional*):\n                A dictionary device identifier to maximum memory. Will default to the maximum memory available for each\n                GPU and the available CPU RAM if unset.\n            offload_folder (`str` or `os.PathLike`, *optional*):\n                If the `device_map` contains any value `"disk"`, the folder where we will offload weights.\n            offload_state_dict (`bool`, *optional*):\n                If `True`, will temporarily offload the CPU state dict to the hard drive to avoid getting out of CPU\n                RAM if the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to\n                `True` when there is some disk offload.\n            load_in_8bit (`bool`, *optional*, defaults to `False`):\n                If `True`, will convert the loaded model into mixed-8bit quantized model. To use this feature please\n                install `bitsandbytes` (`pip install -U bitsandbytes`).\n            load_in_4bit (`bool`, *optional*, defaults to `False`):\n                If `True`, will convert the loaded model into 4bit precision quantized model. To use this feature\n                install the latest version of `bitsandbytes` (`pip install -U bitsandbytes`).\n            quantization_config (`Union[QuantizationConfigMixin,Dict]`, *optional*):\n                A dictionary of configuration parameters or a QuantizationConfigMixin object for quantization (e.g\n                bitsandbytes, gptq)\n            subfolder (`str`, *optional*, defaults to `""`):\n                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can\n                specify the folder name here.\n            variant (`str`, *optional*):\n                If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. `variant` is\n                ignored when using `from_tf` or `from_flax`.\n            use_safetensors (`bool`, *optional*, defaults to `None`):\n                Whether or not to use `safetensors` checkpoints. Defaults to `None`. If not specified and `safetensors`\n                is not installed, it will be set to `False`.\n\n            kwargs (remaining dictionary of keyword arguments, *optional*):\n                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,\n                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or\n                automatically loaded:\n\n                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the\n                      underlying model\'s `__init__` method (we assume all relevant updates to the configuration have\n                      already been done)\n                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class\n                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that\n                      corresponds to a configuration attribute will be used to override said attribute with the\n                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute\n                      will be passed to the underlying model\'s `__init__` function.\n\n        <Tip>\n\n        Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to\n        use this method in a firewalled environment.\n\n        </Tip>\n\n        Examples:\n\n        ```python\n        >>> from transformers import BertConfig, BertModel\n\n        >>> # Download model and configuration from huggingface.co and cache.\n        >>> model = BertModel.from_pretrained("bert-base-uncased")\n        >>> # Model was saved using *save_pretrained(\'./test/saved_model/\')* (for example purposes, not runnable).\n        >>> model = BertModel.from_pretrained("./test/saved_model/")\n        >>> # Update configuration during loading.\n        >>> model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)\n        >>> assert model.config.output_attentions == True\n        >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).\n        >>> config = BertConfig.from_json_file("./tf_model/my_tf_model_config.json")\n        >>> model = BertModel.from_pretrained("./tf_model/my_tf_checkpoint.ckpt.index", from_tf=True, config=config)\n        >>> # Loading from a Flax checkpoint file instead of a PyTorch model (slower)\n        >>> model = BertModel.from_pretrained("bert-base-uncased", from_flax=True)\n        ```\n\n        * `low_cpu_mem_usage` algorithm:\n\n        This is an experimental function that loads the model using ~1x model size CPU memory\n\n        Here is how it works:\n\n        1. save which state_dict keys we have\n        2. drop state_dict before the model is created, since the latter takes 1x model size CPU memory\n        3. after the model has been instantiated switch to the meta device all params/buffers that\n        are going to be replaced from the loaded state_dict\n        4. load state_dict 2nd time\n        5. replace the params/buffers from the state_dict\n\n        Currently, it can\'t handle deepspeed ZeRO stage 3 and ignores loading errors\n\n        '
        state_dict = kwargs.pop('state_dict', None)
        from_tf = kwargs.pop('from_tf', False)
        from_flax = kwargs.pop('from_flax', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)
        output_loading_info = kwargs.pop('output_loading_info', False)
        use_auth_token = kwargs.pop('use_auth_token', None)
        trust_remote_code = kwargs.pop('trust_remote_code', None)
        _ = kwargs.pop('mirror', None)
        from_pipeline = kwargs.pop('_from_pipeline', None)
        from_auto_class = kwargs.pop('_from_auto', False)
        _fast_init = kwargs.pop('_fast_init', True)
        torch_dtype = kwargs.pop('torch_dtype', None)
        low_cpu_mem_usage = kwargs.pop('low_cpu_mem_usage', None)
        device_map = kwargs.pop('device_map', None)
        max_memory = kwargs.pop('max_memory', None)
        offload_folder = kwargs.pop('offload_folder', None)
        offload_state_dict = kwargs.pop('offload_state_dict', False)
        load_in_8bit = kwargs.pop('load_in_8bit', False)
        load_in_4bit = kwargs.pop('load_in_4bit', False)
        quantization_config = kwargs.pop('quantization_config', None)
        subfolder = kwargs.pop('subfolder', '')
        commit_hash = kwargs.pop('_commit_hash', None)
        variant = kwargs.pop('variant', None)
        adapter_kwargs = kwargs.pop('adapter_kwargs', {})
        adapter_name = kwargs.pop('adapter_name', 'default')
        use_flash_attention_2 = kwargs.pop('use_flash_attention_2', False)
        if is_fsdp_enabled():
            low_cpu_mem_usage = True
        if use_auth_token is not None:
            warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
            if token is not None:
                raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
            token = use_auth_token
        if token is not None and adapter_kwargs is not None and ('token' not in adapter_kwargs):
            adapter_kwargs['token'] = token
        if use_safetensors is None and (not is_safetensors_available()):
            use_safetensors = False
        if is_bitsandbytes_available():
            is_8bit_serializable = version.parse(importlib.metadata.version('bitsandbytes')) > version.parse('0.37.2')
        else:
            is_8bit_serializable = False
        if trust_remote_code is True:
            logger.warning('The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.')
        if commit_hash is None:
            if not isinstance(config, PretrainedConfig):
                resolved_config_file = cached_file(pretrained_model_name_or_path, CONFIG_NAME, cache_dir=cache_dir, force_download=force_download, resume_download=resume_download, proxies=proxies, local_files_only=local_files_only, token=token, revision=revision, subfolder=subfolder, _raise_exceptions_for_missing_entries=False, _raise_exceptions_for_connection_errors=False)
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            else:
                commit_hash = getattr(config, '_commit_hash', None)
        if is_peft_available():
            _adapter_model_path = adapter_kwargs.pop('_adapter_model_path', None)
            if _adapter_model_path is None:
                _adapter_model_path = find_adapter_config_file(pretrained_model_name_or_path, cache_dir=cache_dir, force_download=force_download, resume_download=resume_download, proxies=proxies, local_files_only=local_files_only, _commit_hash=commit_hash, **adapter_kwargs)
            if _adapter_model_path is not None and os.path.isfile(_adapter_model_path):
                with open(_adapter_model_path, 'r', encoding='utf-8') as f:
                    _adapter_model_path = pretrained_model_name_or_path
                    pretrained_model_name_or_path = json.load(f)['base_model_name_or_path']
        else:
            _adapter_model_path = None
        if isinstance(device_map, torch.device):
            device_map = {'': device_map}
        elif isinstance(device_map, str) and device_map not in ['auto', 'balanced', 'balanced_low_0', 'sequential']:
            try:
                device_map = {'': torch.device(device_map)}
            except RuntimeError:
                raise ValueError(f"When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or 'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}.")
        elif isinstance(device_map, int):
            if device_map < 0:
                raise ValueError("You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' ")
            else:
                device_map = {'': device_map}
        if device_map is not None:
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
            elif not low_cpu_mem_usage:
                raise ValueError('Passing along a `device_map` requires `low_cpu_mem_usage=True`')
        if low_cpu_mem_usage:
            if device_map is not None:
                require_version_core('torch>=1.10')
            if is_deepspeed_zero3_enabled():
                raise ValueError('DeepSpeed Zero-3 is not compatible with `low_cpu_mem_usage=True` or with passing a `device_map`.')
            elif not is_accelerate_available():
                raise ImportError('Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: `pip install accelerate`')
        quantization_method_from_args = None
        if quantization_config is not None:
            quantization_method_from_args = getattr(quantization_config, 'quant_method', QuantizationMethod.BITS_AND_BYTES)
            if quantization_method_from_args == QuantizationMethod.AWQ:
                raise ValueError('You cannot pass an `AwqConfig` when loading a model as you can only use AWQ models for inference. To quantize transformers models with AWQ algorithm, please refer to our quantization docs: https://huggingface.co/docs/transformers/main_classes/quantization ')
        if quantization_config is None and (load_in_8bit or load_in_4bit):
            quantization_method_from_args = QuantizationMethod.BITS_AND_BYTES
            (quantization_config, kwargs) = BitsAndBytesConfig.from_dict(config_dict={'load_in_8bit': load_in_8bit, 'load_in_4bit': load_in_4bit}, return_unused_kwargs=True, **kwargs)
        elif quantization_method_from_args == QuantizationMethod.BITS_AND_BYTES:
            load_in_8bit = quantization_config.load_in_8bit
            load_in_4bit = quantization_config.load_in_4bit
            quantization_config_kwargs = {k: v for (k, v) in kwargs.items() if k in inspect.signature(BitsAndBytesConfig).parameters}
            if len(quantization_config_kwargs) > 0:
                raise ValueError("You can't pass `load_in_8bit` or any other `BitsAndBytesConfig` argument as a kwarg when passing `quantization_config` argument at the same time.")
        if load_in_8bit or load_in_4bit:
            if not (is_accelerate_available() and is_bitsandbytes_available()):
                raise ImportError('Using `load_in_8bit=True` requires Accelerate: `pip install accelerate` and the latest version of bitsandbytes `pip install -i https://test.pypi.org/simple/ bitsandbytes` or pip install bitsandbytes` ')
            if torch_dtype is None:
                logger.info(f'Overriding torch_dtype={torch_dtype} with `torch_dtype=torch.float16` due to requirements of `bitsandbytes` to enable model loading in 8-bit or 4-bit. Pass your own torch_dtype to specify the dtype of the remaining non-linear layers or pass torch_dtype=torch.float16 to remove this warning.')
                torch_dtype = torch.float16
            if device_map is None:
                if torch.cuda.is_available():
                    device_map = {'': torch.cuda.current_device()}
                else:
                    raise RuntimeError('No GPU found. A GPU is needed for quantization.')
                logger.info("The device_map was not initialized. Setting device_map to {'':torch.cuda.current_device()}. If you want to use the model for inference, please set device_map ='auto' ")
                if low_cpu_mem_usage is None:
                    low_cpu_mem_usage = True
            if from_tf or from_flax:
                raise ValueError('Converting into 4-bit or 8-bit weights from tf/flax weights is currently not supported, please make sure the weights are in PyTorch format.')
        user_agent = {'file_type': 'model', 'framework': 'pytorch', 'from_auto_class': from_auto_class}
        if from_pipeline is not None:
            user_agent['using_pipeline'] = from_pipeline
        if is_offline_mode() and (not local_files_only):
            logger.info('Offline mode: forcing local_files_only=True')
            local_files_only = True
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            (config, model_kwargs) = cls.config_class.from_pretrained(config_path, cache_dir=cache_dir, return_unused_kwargs=True, force_download=force_download, resume_download=resume_download, proxies=proxies, local_files_only=local_files_only, token=token, revision=revision, subfolder=subfolder, _from_auto=from_auto_class, _from_pipeline=from_pipeline, **kwargs)
        else:
            model_kwargs = kwargs
        quantizer = None
        quantization_method_from_config = None
        if hasattr(config, 'quantization_config'):
            quantization_method_from_config = config.quantization_config.get('quant_method', QuantizationMethod.BITS_AND_BYTES)
        if quantization_method_from_config == QuantizationMethod.GPTQ and quantization_method_from_args is not None:
            loading_attr_dict = quantization_config.get_loading_attributes()
            for (attr, val) in loading_attr_dict.items():
                config.quantization_config[attr] = val
            quantization_method_from_args = None
            logger.warning("You passed `quantization_config` to `from_pretrained` but the model you're loading already has a `quantization_config` attribute and has already quantized weights. However, loading attributes (e.g. use_exllama, exllama_config, use_cuda_fp16, max_input_length) will be overwritten with the one you passed to `from_pretrained`. The rest will be ignored.")
        if quantization_method_from_args == QuantizationMethod.GPTQ or quantization_method_from_config == QuantizationMethod.GPTQ:
            gptq_supports_cpu = version.parse(importlib.metadata.version('auto-gptq')) > version.parse('0.4.2')
            if not gptq_supports_cpu and (not torch.cuda.is_available()):
                raise RuntimeError('GPU is required to quantize or run quantize model.')
            elif not (is_optimum_available() and is_auto_gptq_available()):
                raise ImportError('Loading a GPTQ quantized model requires optimum (`pip install optimum`) and auto-gptq library (`pip install auto-gptq`)')
            elif version.parse(importlib.metadata.version('auto_gptq')) < version.parse('0.4.2'):
                raise ImportError('You need a version of auto_gptq >= 0.4.2 to use GPTQ: `pip install --upgrade auto-gptq`')
            else:
                from optimum.gptq import GPTQQuantizer
            if quantization_method_from_config == QuantizationMethod.GPTQ:
                quantization_config = GPTQConfig.from_dict(config.quantization_config)
                config.quantization_config = quantization_config
            if torch_dtype is None:
                torch_dtype = torch.float16
            else:
                logger.info('We suggest you to set `torch_dtype=torch.float16` for better efficiency with GPTQ.')
            quantizer = GPTQQuantizer.from_dict(quantization_config.to_dict_optimum())
        elif quantization_method_from_config == QuantizationMethod.AWQ:
            if not torch.cuda.is_available():
                raise RuntimeError('GPU is required to run AWQ quantized model.')
            if not is_auto_awq_available():
                raise ImportError('Loading an AWQ quantized model requires auto-awq library (`pip install autoawq`)')
            if not is_accelerate_available():
                raise ImportError('Loading an AWQ quantized model requires accelerate (`pip install accelerate`)')
            if device_map is None:
                logger.warning('You have loaded an AWQ model on CPU and have a CUDA device available, make sure to set your model on a GPU device in order to run your model.')
            elif device_map is not None:
                if isinstance(device_map, dict) and ('cpu' in device_map.values() or 'disk' in device_map.values()):
                    raise ValueError('You are attempting to load an AWQ model with a device_map that contains a CPU or disk device. This is not supported. Please remove the CPU or disk device from the device_map.')
            if torch_dtype is None:
                torch_dtype = torch.float16
            else:
                logger.info('We suggest you to set `torch_dtype=torch.float16` for better efficiency with AWQ.')
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
        if is_8bit_serializable and quantization_method_from_args == QuantizationMethod.BITS_AND_BYTES and load_in_8bit:
            if quantization_method_from_config == QuantizationMethod.BITS_AND_BYTES:
                logger.warning("You passed `quantization_config` to `from_pretrained` but the model you're loading already has a `quantization_config` attribute. The `quantization_config` attribute will be overwritten with the one you passed to `from_pretrained`.")
            config.quantization_config = quantization_config
        elif is_8bit_serializable and (not load_in_8bit) and (quantization_method_from_config == QuantizationMethod.BITS_AND_BYTES):
            quantization_config = config.quantization_config
            if isinstance(quantization_config, dict):
                quantization_config = BitsAndBytesConfig.from_dict(quantization_config, return_unused_kwargs=False)
            elif isinstance(quantization_config, BitsAndBytesConfig):
                pass
            else:
                raise ValueError(f'Invalid type for `quantization_config`: {type(quantization_config)}. Should be a `dict` or a `BitsAndBytesConfig` instance.')
            load_in_8bit = quantization_config.load_in_8bit
            if load_in_8bit:
                if torch_dtype is None:
                    torch_dtype = torch.float16
                if device_map is None:
                    if torch.cuda.is_available():
                        device_map = {'': torch.cuda.current_device()}
                    else:
                        raise RuntimeError('No GPU found. A GPU is needed for quantization.')
                    logger.info("The device_map was not initialized. Setting device_map to {'':torch.cuda.current_device()}. If you want to use the model for inference, please set device_map ='auto' ")
                    if low_cpu_mem_usage is None:
                        low_cpu_mem_usage = True
        elif not is_8bit_serializable and (not load_in_8bit) and (quantization_method_from_config == QuantizationMethod.BITS_AND_BYTES):
            logger.warning("Detected the presence of a `quantization_config` attribute in the model's configuration but you don't have the correct `bitsandbytes` version to support int8 serialization. Please install the latest version of `bitsandbytes` with  `pip install --upgrade bitsandbytes`.")
        is_sharded = False
        sharded_metadata = None
        loading_info = None
        keep_in_fp32_modules = None
        use_keep_in_fp32_modules = False
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)
            if is_local:
                if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + '.index')):
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + '.index')
                elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
                elif from_flax and os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
                elif use_safetensors is not False and os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))):
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))
                elif use_safetensors is not False and os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))):
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
                    is_sharded = True
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant))):
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant))
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant))):
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant))
                    is_sharded = True
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + '.index')) or os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)):
                    raise EnvironmentError(f'Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory {pretrained_model_name_or_path} but there is a file for TensorFlow weights. Use `from_tf=True` to load this model from those weights.')
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)):
                    raise EnvironmentError(f'Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory {pretrained_model_name_or_path} but there is a file for Flax weights. Use `from_flax=True` to load this model from those weights.')
                elif use_safetensors:
                    raise EnvironmentError(f'Error no file named {_add_variant(SAFE_WEIGHTS_NAME, variant)} found in directory {pretrained_model_name_or_path}.')
                else:
                    raise EnvironmentError(f"Error no file named {_add_variant(WEIGHTS_NAME, variant)}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME + '.index'} or {FLAX_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path}.")
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
                archive_file = pretrained_model_name_or_path
                is_local = True
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path + '.index')):
                if not from_tf:
                    raise ValueError(f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, please set from_tf to True to load from this checkpoint.")
                archive_file = os.path.join(subfolder, pretrained_model_name_or_path + '.index')
                is_local = True
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = download_url(pretrained_model_name_or_path)
            else:
                if from_tf:
                    filename = TF2_WEIGHTS_NAME
                elif from_flax:
                    filename = FLAX_WEIGHTS_NAME
                elif use_safetensors is not False:
                    filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
                else:
                    filename = _add_variant(WEIGHTS_NAME, variant)
                try:
                    cached_file_kwargs = {'cache_dir': cache_dir, 'force_download': force_download, 'proxies': proxies, 'resume_download': resume_download, 'local_files_only': local_files_only, 'token': token, 'user_agent': user_agent, 'revision': revision, 'subfolder': subfolder, '_raise_exceptions_for_missing_entries': False, '_commit_hash': commit_hash}
                    resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)
                    if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                        resolved_archive_file = cached_file(pretrained_model_name_or_path, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant), **cached_file_kwargs)
                        if resolved_archive_file is not None:
                            is_sharded = True
                        elif use_safetensors:
                            raise EnvironmentError(f' {_add_variant(SAFE_WEIGHTS_NAME, variant)} or {_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} and thus cannot be loaded with `safetensors`. Please make sure that the model has been saved with `safe_serialization=True` or do not set `use_safetensors=True`.')
                        else:
                            filename = _add_variant(WEIGHTS_NAME, variant)
                            resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)
                    if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                        resolved_archive_file = cached_file(pretrained_model_name_or_path, _add_variant(WEIGHTS_INDEX_NAME, variant), **cached_file_kwargs)
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if resolved_archive_file is None:
                        has_file_kwargs = {'revision': revision, 'proxies': proxies, 'token': token}
                        if has_file(pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **has_file_kwargs):
                            raise EnvironmentError(f'{pretrained_model_name_or_path} does not appear to have a file named {_add_variant(WEIGHTS_NAME, variant)} but there is a file for TensorFlow weights. Use `from_tf=True` to load this model from those weights.')
                        elif has_file(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME, **has_file_kwargs):
                            raise EnvironmentError(f'{pretrained_model_name_or_path} does not appear to have a file named {_add_variant(WEIGHTS_NAME, variant)} but there is a file for Flax weights. Use `from_flax=True` to load this model from those weights.')
                        elif variant is not None and has_file(pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs):
                            raise EnvironmentError(f'{pretrained_model_name_or_path} does not appear to have a file named {_add_variant(WEIGHTS_NAME, variant)} but there is a file without the variant {variant}. Use `variant=None` to load this model from those weights.')
                        else:
                            raise EnvironmentError(f'{pretrained_model_name_or_path} does not appear to have a file named {_add_variant(WEIGHTS_NAME, variant)}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}.')
                except EnvironmentError:
                    raise
                except Exception:
                    raise EnvironmentError(f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named {_add_variant(WEIGHTS_NAME, variant)}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}.")
            if is_local:
                logger.info(f'loading weights file {archive_file}')
                resolved_archive_file = archive_file
            else:
                logger.info(f'loading weights file {filename} from cache at {resolved_archive_file}')
        else:
            resolved_archive_file = None
        if is_sharded:
            (resolved_archive_file, sharded_metadata) = get_checkpoint_shard_files(pretrained_model_name_or_path, resolved_archive_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only, token=token, user_agent=user_agent, revision=revision, subfolder=subfolder, _commit_hash=commit_hash)
        if is_safetensors_available() and isinstance(resolved_archive_file, str) and resolved_archive_file.endswith('.safetensors'):
            with safe_open(resolved_archive_file, framework='pt') as f:
                metadata = f.metadata()
            if metadata.get('format') == 'pt':
                pass
            elif metadata.get('format') == 'tf':
                from_tf = True
                logger.info('A TensorFlow safetensors file is being loaded in a PyTorch model.')
            elif metadata.get('format') == 'flax':
                from_flax = True
                logger.info('A Flax safetensors file is being loaded in a PyTorch model.')
            else:
                raise ValueError(f"Incompatible safetensors file. File metadata is not ['pt', 'tf', 'flax'] but {metadata.get('format')}")
        from_pt = not from_tf | from_flax
        if from_pt:
            if not is_sharded and state_dict is None:
                state_dict = load_state_dict(resolved_archive_file)
            dtype_orig = None
            if torch_dtype is not None:
                if isinstance(torch_dtype, str):
                    if torch_dtype == 'auto':
                        if hasattr(config, 'torch_dtype') and config.torch_dtype is not None:
                            torch_dtype = config.torch_dtype
                            logger.info(f"Will use torch_dtype={torch_dtype} as defined in model's config object")
                        else:
                            if is_sharded and 'dtype' in sharded_metadata:
                                torch_dtype = sharded_metadata['dtype']
                            elif not is_sharded:
                                torch_dtype = get_state_dict_dtype(state_dict)
                            else:
                                one_state_dict = load_state_dict(resolved_archive_file[0])
                                torch_dtype = get_state_dict_dtype(one_state_dict)
                                del one_state_dict
                            logger.info("Since the `torch_dtype` attribute can't be found in model's config object, will use torch_dtype={torch_dtype} as derived from model's weights")
                    else:
                        raise ValueError(f'`torch_dtype` can be either `torch.dtype` or `"auto"`, but received {torch_dtype}')
                dtype_orig = cls._set_default_torch_dtype(torch_dtype)
            use_keep_in_fp32_modules = cls._keep_in_fp32_modules is not None and (torch_dtype == torch.float16 or load_in_4bit or load_in_8bit)
            if is_sharded:
                loaded_state_dict_keys = sharded_metadata['all_checkpoint_keys']
            else:
                loaded_state_dict_keys = list(state_dict.keys())
            if low_cpu_mem_usage or (use_keep_in_fp32_modules and is_accelerate_available()):
                state_dict = None
        config.name_or_path = pretrained_model_name_or_path
        init_contexts = [no_init_weights(_enable=_fast_init)]
        if is_deepspeed_zero3_enabled():
            import deepspeed
            logger.info('Detected DeepSpeed ZeRO-3: activating zero.init() for this model')
            init_contexts = [deepspeed.zero.Init(config_dict_or_path=deepspeed_config())] + init_contexts
        elif load_in_8bit or load_in_4bit or low_cpu_mem_usage:
            init_contexts.append(init_empty_weights())
        if use_flash_attention_2:
            config = cls._check_and_enable_flash_attn_2(config, torch_dtype=torch_dtype, device_map=device_map)
        with ContextManagers(init_contexts):
            model = cls(config, *model_args, **model_kwargs)
        config = model.config
        if use_keep_in_fp32_modules:
            if is_accelerate_available():
                low_cpu_mem_usage = True
            keep_in_fp32_modules = model._keep_in_fp32_modules
        else:
            keep_in_fp32_modules = []
        if load_in_8bit or load_in_4bit:
            from .integrations import get_keys_to_not_convert, replace_with_bnb_linear
            llm_int8_skip_modules = quantization_config.llm_int8_skip_modules
            load_in_8bit_fp32_cpu_offload = quantization_config.llm_int8_enable_fp32_cpu_offload
            if load_in_8bit:
                logger.info('Detected 8-bit loading: activating 8-bit loading for this model')
            else:
                logger.info('Detected 4-bit loading: activating 4-bit loading for this model')
            if llm_int8_skip_modules is None:
                modules_to_not_convert = get_keys_to_not_convert(model)
            else:
                modules_to_not_convert = llm_int8_skip_modules
            if not isinstance(modules_to_not_convert, list):
                modules_to_not_convert = [modules_to_not_convert]
            modules_to_not_convert.extend(keep_in_fp32_modules)
            if isinstance(device_map, dict) and len(device_map.keys()) > 1:
                keys_on_cpu = [key for (key, value) in device_map.items() if value in ['disk', 'cpu']]
                if len(keys_on_cpu) > 0 and (not load_in_8bit_fp32_cpu_offload):
                    raise ValueError('If you want to offload some keys to `cpu` or `disk`, you need to set `llm_int8_enable_fp32_cpu_offload=True`. Note that these modules will not be  converted to 8-bit but kept in 32-bit.')
                modules_to_not_convert.extend(keys_on_cpu)
            supports_4bit = version.parse(importlib.metadata.version('bitsandbytes')) >= version.parse('0.39.0')
            if load_in_4bit and (not supports_4bit):
                raise ValueError('You have a version of `bitsandbytes` that is not compatible with 4bit inference and training make sure you have the latest version of `bitsandbytes` installed')
            model = replace_with_bnb_linear(model, modules_to_not_convert=modules_to_not_convert, quantization_config=quantization_config)
            model._is_quantized_training_enabled = version.parse(importlib.metadata.version('bitsandbytes')) >= version.parse('0.37.0')
            config.quantization_config = quantization_config
            model.is_8bit_serializable = is_8bit_serializable
        if load_in_8bit and torch_dtype is None:
            logger.warning('You are loading your model in 8bit but you did not specify a `torch_dtype` attribute. All non-linear modules will be loaded in full precision. If you want to load the other modules in other precision, please specify a `torch_dtype` attribute.')
        if quantization_method_from_config == QuantizationMethod.GPTQ:
            model = quantizer.convert_model(model)
            model._is_quantized_training_enabled = True
        elif quantization_method_from_config == QuantizationMethod.AWQ:
            from .integrations import get_keys_to_not_convert, replace_with_awq_linear
            modules_to_not_convert = get_keys_to_not_convert(model)
            if quantization_config is None:
                quantization_config = AwqConfig.from_dict(config.quantization_config)
            (model, has_been_replaced) = replace_with_awq_linear(model, quantization_config=quantization_config, modules_to_not_convert=modules_to_not_convert)
            model._is_quantized_training_enabled = False
            if not has_been_replaced:
                logger.warning('You are loading an AWQ model but no linear modules were found in your model. Please double check your model architecture, or submit an issue on github if you think this is a bug.')
        if quantization_method_from_config is not None:
            model.quantization_method = quantization_method_from_config
        elif quantization_method_from_args is not None:
            model.quantization_method = quantization_method_from_args
        if hasattr(model, 'quantization_method'):
            model.is_quantized = True
            config._pre_quantization_dtype = torch_dtype
        if isinstance(device_map, str):
            special_dtypes = {}
            if load_in_8bit or load_in_4bit:
                special_dtypes.update({name: torch_dtype for (name, _) in model.named_parameters() if any((m in name for m in modules_to_not_convert))})
            special_dtypes.update({name: torch.float32 for (name, _) in model.named_parameters() if any((m in name for m in keep_in_fp32_modules))})
            target_dtype = torch_dtype
            if load_in_4bit:
                if version.parse(importlib.metadata.version('accelerate')) > version.parse('0.19.0'):
                    from accelerate.utils import CustomDtype
                    target_dtype = CustomDtype.INT4
                else:
                    raise ValueError("You are using `device_map='auto'` on a 4bit loaded version of the model. To automatically compute the appropriate device map, you should upgrade your `accelerate` library, `pip install --upgrade accelerate` or install it from source to support fp4 auto device map calculation. You may encounter unexpected behavior, or pass your own device map")
            elif load_in_8bit:
                target_dtype = torch.int8
            no_split_modules = model._get_no_split_modules(device_map)
            if device_map not in ['auto', 'balanced', 'balanced_low_0', 'sequential']:
                raise ValueError("If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or 'sequential'.")
            device_map_kwargs = {'no_split_module_classes': no_split_modules}
            if 'special_dtypes' in inspect.signature(infer_auto_device_map).parameters:
                device_map_kwargs['special_dtypes'] = special_dtypes
            elif len(special_dtypes) > 0:
                logger.warning('This model has some weights that should be kept in higher precision, you need to upgrade `accelerate` to properly deal with them (`pip install --upgrade accelerate`).')
            if device_map != 'sequential':
                max_memory = get_balanced_memory(model, dtype=target_dtype, low_zero=device_map == 'balanced_low_0', max_memory=max_memory, **device_map_kwargs)
            else:
                max_memory = get_max_memory(max_memory)
            if getattr(model, 'quantization_method', None) == QuantizationMethod.BITS_AND_BYTES:
                max_memory = {key: val * 0.9 for (key, val) in max_memory.items()}
            device_map_kwargs['max_memory'] = max_memory
            model.tie_weights()
            device_map = infer_auto_device_map(model, dtype=target_dtype, **device_map_kwargs)
            if load_in_8bit or load_in_4bit:
                device_map_without_lm_head = {key: device_map[key] for key in device_map.keys() if key not in modules_to_not_convert}
                if 'cpu' in device_map_without_lm_head.values() or 'disk' in device_map_without_lm_head.values():
                    raise ValueError('\n                        Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit\n                        the quantized model. If you want to dispatch the model on the CPU or the disk while keeping\n                        these modules in 32-bit, you need to set `load_in_8bit_fp32_cpu_offload=True` and pass a custom\n                        `device_map` to `from_pretrained`. Check\n                        https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu\n                        for more details.\n                        ')
                del device_map_without_lm_head
        elif device_map is not None:
            model.tie_weights()
            tied_params = find_tied_parameters(model)
            check_tied_parameters_on_same_device(tied_params, device_map)
        if from_tf:
            if resolved_archive_file.endswith('.index'):
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])
            else:
                try:
                    from .modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model
                    (model, loading_info) = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True, output_loading_info=True)
                except ImportError:
                    logger.error('Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.')
                    raise
        elif from_flax:
            try:
                from .modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model
                model = load_flax_checkpoint_in_pytorch_model(model, resolved_archive_file)
            except ImportError:
                logger.error('Loading a Flax model in PyTorch, requires both PyTorch and Flax to be installed. Please see https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for installation instructions.')
                raise
        elif from_pt:
            if dtype_orig is not None:
                torch.set_default_dtype(dtype_orig)
            (model, missing_keys, unexpected_keys, mismatched_keys, offload_index, error_msgs) = cls._load_pretrained_model(model, state_dict, loaded_state_dict_keys, resolved_archive_file, pretrained_model_name_or_path, ignore_mismatched_sizes=ignore_mismatched_sizes, sharded_metadata=sharded_metadata, _fast_init=_fast_init, low_cpu_mem_usage=low_cpu_mem_usage, device_map=device_map, offload_folder=offload_folder, offload_state_dict=offload_state_dict, dtype=torch_dtype, is_quantized=getattr(model, 'quantization_method', None) == QuantizationMethod.BITS_AND_BYTES, keep_in_fp32_modules=keep_in_fp32_modules)
        model.is_loaded_in_4bit = load_in_4bit
        model.is_loaded_in_8bit = load_in_8bit
        model.tie_weights()
        model.eval()
        if model.can_generate() and pretrained_model_name_or_path is not None:
            try:
                model.generation_config = GenerationConfig.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir, force_download=force_download, resume_download=resume_download, proxies=proxies, local_files_only=local_files_only, token=token, revision=revision, subfolder=subfolder, _from_auto=from_auto_class, _from_pipeline=from_pipeline, **kwargs)
            except OSError:
                logger.info('Generation config file not found, using a generation config created from the model config.')
                pass
        if device_map is not None:
            device_map_kwargs = {'device_map': device_map, 'offload_dir': offload_folder, 'offload_index': offload_index}
            if 'skip_keys' in inspect.signature(dispatch_model).parameters:
                device_map_kwargs['skip_keys'] = model._skip_keys_device_placement
            dispatch_model(model, **device_map_kwargs)
        if quantization_method_from_args == QuantizationMethod.GPTQ:
            if quantization_config.tokenizer is None:
                quantization_config.tokenizer = pretrained_model_name_or_path
            if cls.main_input_name != 'input_ids':
                raise RuntimeError('We can only quantize pure text model.')
            quantizer.quantize_model(model, quantization_config.tokenizer)
            config.quantization_config = GPTQConfig.from_dict_optimum(quantizer.to_dict())
            model._is_quantized_training_enabled = True
        if quantization_method_from_config == QuantizationMethod.GPTQ:
            model = quantizer.post_init_model(model)
        if _adapter_model_path is not None:
            model.load_adapter(_adapter_model_path, adapter_name=adapter_name, token=token, adapter_kwargs=adapter_kwargs)
        if output_loading_info:
            if loading_info is None:
                loading_info = {'missing_keys': missing_keys, 'unexpected_keys': unexpected_keys, 'mismatched_keys': mismatched_keys, 'error_msgs': error_msgs}
            return (model, loading_info)
        return model

    @classmethod
    def _load_pretrained_model(cls, model, state_dict, loaded_keys, resolved_archive_file, pretrained_model_name_or_path, ignore_mismatched_sizes=False, sharded_metadata=None, _fast_init=True, low_cpu_mem_usage=False, device_map=None, offload_folder=None, offload_state_dict=None, dtype=None, is_quantized=False, keep_in_fp32_modules=None):
        if False:
            i = 10
            return i + 15
        is_safetensors = False
        if is_quantized:
            from .integrations import set_module_quantized_tensor_to_device
        if device_map is not None and 'disk' in device_map.values():
            archive_file = resolved_archive_file[0] if isinstance(resolved_archive_file, (list, tuple)) else resolved_archive_file
            is_safetensors = archive_file.endswith('.safetensors')
            if offload_folder is None and (not is_safetensors):
                raise ValueError('The current `device_map` had weights offloaded to the disk. Please provide an `offload_folder` for them. Alternatively, make sure you have `safetensors` installed if the model you are using offers the weights in this format.')
            if offload_folder is not None:
                os.makedirs(offload_folder, exist_ok=True)
            if offload_state_dict is None:
                offload_state_dict = True
        is_sharded_safetensors = is_safetensors and sharded_metadata is not None
        model.tie_weights()
        model_state_dict = model.state_dict()
        expected_keys = list(model_state_dict.keys())
        prefix = model.base_model_prefix

        def _fix_key(key):
            if False:
                i = 10
                return i + 15
            if 'beta' in key:
                return key.replace('beta', 'bias')
            if 'gamma' in key:
                return key.replace('gamma', 'weight')
            return key
        original_loaded_keys = loaded_keys
        loaded_keys = [_fix_key(key) for key in loaded_keys]
        if len(prefix) > 0:
            has_prefix_module = any((s.startswith(prefix) for s in loaded_keys))
            expects_prefix_module = any((s.startswith(prefix) for s in expected_keys))
        else:
            has_prefix_module = False
            expects_prefix_module = False
        remove_prefix_from_model = not has_prefix_module and expects_prefix_module
        add_prefix_to_model = has_prefix_module and (not expects_prefix_module)
        if remove_prefix_from_model:
            _prefix = f'{prefix}.'
            expected_keys_not_prefixed = [s for s in expected_keys if not s.startswith(_prefix)]
            expected_keys = [s[len(_prefix):] if s.startswith(_prefix) else s for s in expected_keys]
        elif add_prefix_to_model:
            expected_keys = ['.'.join([prefix, s]) for s in expected_keys]
        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = set(loaded_keys) - set(expected_keys)
        model_buffers = {n for (n, _) in model.named_buffers()}
        if remove_prefix_from_model:
            model_buffers = {key[len(_prefix):] if key.startswith(_prefix) else key for key in model_buffers}
        elif add_prefix_to_model:
            model_buffers = {'.'.join([prefix, key]) for key in model_buffers}
        unexpected_keys = list(unexpected_keys - model_buffers)
        model.tie_weights()
        if device_map is None and (not is_fsdp_enabled()):
            ptrs = collections.defaultdict(list)
            for (name, tensor) in model.state_dict().items():
                id_tensor = id_tensor_storage(tensor)
                ptrs[id_tensor].append(name)
            tied_params = [names for (_, names) in ptrs.items() if len(names) > 1]
        else:
            tied_params = find_tied_parameters(model)
        for group in tied_params:
            if remove_prefix_from_model:
                group = [key[len(_prefix):] if key.startswith(_prefix) else key for key in group]
            elif add_prefix_to_model:
                group = ['.'.join([prefix, key]) for key in group]
            missing_in_group = [k for k in missing_keys if k in group]
            if len(missing_in_group) > 0 and len(missing_in_group) < len(group):
                missing_keys = [k for k in missing_keys if k not in missing_in_group]
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]
        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]
        if low_cpu_mem_usage:
            for key in missing_keys:
                if key in list(model_state_dict.keys()):
                    key = key
                elif f'{prefix}.{key}' in list(model_state_dict.keys()):
                    key = f'{prefix}.{key}'
                elif key.startswith(prefix) and '.'.join(key.split('.')[1:]) in list(model_state_dict.keys()):
                    key = '.'.join(key.split('.')[1:])
                param = model_state_dict[key]
                target_dtype = dtype
                if keep_in_fp32_modules is not None and dtype == torch.float16 and any((module_to_keep_in_fp32 in key.split('.') for module_to_keep_in_fp32 in keep_in_fp32_modules)):
                    target_dtype = torch.float32
                if param.device == torch.device('meta'):
                    if not is_quantized:
                        set_module_tensor_to_device(model, key, 'cpu', torch.empty(*param.size(), dtype=target_dtype))
                    else:
                        set_module_quantized_tensor_to_device(model, key, 'cpu', torch.empty(*param.size(), dtype=target_dtype))
        if _fast_init:
            if remove_prefix_from_model:
                _loaded_keys = [f'{prefix}.{k}' for k in loaded_keys]
            elif add_prefix_to_model:
                _loaded_keys = [k[len(prefix) + 1:] for k in loaded_keys]
            else:
                _loaded_keys = loaded_keys
            set_initialized_submodules(model, _loaded_keys)
            model.apply(model._initialize_weights)
        if keep_in_fp32_modules is not None:
            for (name, param) in model.named_parameters():
                if any((module_to_keep_in_fp32 in name.split('.') for module_to_keep_in_fp32 in keep_in_fp32_modules)):
                    param.data = param.data.to(torch.float32)
        start_prefix = ''
        model_to_load = model
        if len(cls.base_model_prefix) > 0 and (not hasattr(model, cls.base_model_prefix)) and has_prefix_module:
            start_prefix = cls.base_model_prefix + '.'
        if len(cls.base_model_prefix) > 0 and hasattr(model, cls.base_model_prefix) and (not has_prefix_module):
            model_to_load = getattr(model, cls.base_model_prefix)
            base_model_expected_keys = list(model_to_load.state_dict().keys())
            if any((key in expected_keys_not_prefixed and key not in base_model_expected_keys for key in loaded_keys)):
                raise ValueError('The state dictionary of the model you are trying to load is corrupted. Are you sure it was properly saved?')
            if device_map is not None:
                device_map = {k.replace(f'{cls.base_model_prefix}.', ''): v for (k, v) in device_map.items()}

        def _find_mismatched_keys(state_dict, model_state_dict, loaded_keys, add_prefix_to_model, remove_prefix_from_model, ignore_mismatched_sizes):
            if False:
                return 10
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    if checkpoint_key not in state_dict:
                        continue
                    model_key = checkpoint_key
                    if remove_prefix_from_model:
                        model_key = f'{prefix}.{checkpoint_key}'
                    elif add_prefix_to_model:
                        model_key = '.'.join(checkpoint_key.split('.')[1:])
                    if model_key in model_state_dict and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape:
                        mismatched_keys.append((checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape))
                        del state_dict[checkpoint_key]
            return mismatched_keys
        if resolved_archive_file is not None:
            folder = os.path.sep.join(resolved_archive_file[0].split(os.path.sep)[:-1])
        else:
            folder = None
        if device_map is not None and is_safetensors:
            param_device_map = expand_device_map(device_map, original_loaded_keys, start_prefix)
            str_dtype = str(dtype).replace('torch.', '') if dtype is not None else 'float32'
            if sharded_metadata is None:
                archive_file = resolved_archive_file[0] if isinstance(resolved_archive_file, (list, tuple)) else resolved_archive_file
                weight_map = {p: archive_file for p in original_loaded_keys}
            else:
                weight_map = {p: os.path.join(folder, f) for (p, f) in sharded_metadata['weight_map'].items()}
            offload_index = {p[len(start_prefix):]: {'safetensors_file': f, 'weight_name': p, 'dtype': str_dtype} for (p, f) in weight_map.items() if p.startswith(start_prefix) and param_device_map[p[len(start_prefix):]] == 'disk'}
        if state_dict is not None:
            mismatched_keys = _find_mismatched_keys(state_dict, model_state_dict, original_loaded_keys, add_prefix_to_model, remove_prefix_from_model, ignore_mismatched_sizes)
            error_msgs = _load_state_dict_into_model(model_to_load, state_dict, start_prefix)
            offload_index = None
        else:
            if not isinstance(resolved_archive_file, list):
                resolved_archive_file = [resolved_archive_file]
            error_msgs = []
            mismatched_keys = []
            if not is_safetensors:
                offload_index = {} if device_map is not None and 'disk' in device_map.values() else None
            if offload_state_dict:
                state_dict_folder = tempfile.mkdtemp()
                state_dict_index = {}
            else:
                state_dict_folder = None
                state_dict_index = None
            if is_sharded_safetensors:
                disk_only_shard_files = get_disk_only_shard_files(device_map, sharded_metadata=sharded_metadata, start_prefix=start_prefix)
                disk_only_shard_files = [os.path.join(folder, f) for f in disk_only_shard_files]
            else:
                disk_only_shard_files = []
            if len(resolved_archive_file) > 1:
                resolved_archive_file = logging.tqdm(resolved_archive_file, desc='Loading checkpoint shards')
            for shard_file in resolved_archive_file:
                if shard_file in disk_only_shard_files:
                    continue
                state_dict = load_state_dict(shard_file)
                mismatched_keys += _find_mismatched_keys(state_dict, model_state_dict, original_loaded_keys, add_prefix_to_model, remove_prefix_from_model, ignore_mismatched_sizes)
                if low_cpu_mem_usage:
                    if not is_fsdp_enabled() or is_fsdp_enabled_and_dist_rank_0():
                        (new_error_msgs, offload_index, state_dict_index) = _load_state_dict_into_meta_model(model_to_load, state_dict, loaded_keys, start_prefix, expected_keys, device_map=device_map, offload_folder=offload_folder, offload_index=offload_index, state_dict_folder=state_dict_folder, state_dict_index=state_dict_index, dtype=dtype, is_quantized=is_quantized, is_safetensors=is_safetensors, keep_in_fp32_modules=keep_in_fp32_modules)
                        error_msgs += new_error_msgs
                    else:
                        for (key, param) in model_to_load.state_dict().items():
                            if param.device == torch.device('meta'):
                                if not is_quantized:
                                    set_module_tensor_to_device(model_to_load, key, 'cpu', torch.empty(*param.size(), dtype=dtype))
                                else:
                                    set_module_quantized_tensor_to_device(model_to_load, key, 'cpu', torch.empty(*param.size(), dtype=dtype))
                else:
                    error_msgs += _load_state_dict_into_model(model_to_load, state_dict, start_prefix)
                del state_dict
                gc.collect()
            if offload_index is not None and len(offload_index) > 0:
                if model != model_to_load:
                    prefix = cls.base_model_prefix
                    if not is_safetensors:
                        for weight_name in offload_index:
                            shutil.move(os.path.join(offload_folder, f'{weight_name}.dat'), os.path.join(offload_folder, f'{prefix}.{weight_name}.dat'))
                    offload_index = {f'{prefix}.{key}': value for (key, value) in offload_index.items()}
                if not is_safetensors:
                    save_offload_index(offload_index, offload_folder)
                    offload_index = None
            if offload_state_dict:
                load_offloaded_weights(model_to_load, state_dict_index, state_dict_folder)
                shutil.rmtree(state_dict_folder)
        if len(error_msgs) > 0:
            error_msg = '\n\t'.join(error_msgs)
            if 'size mismatch' in error_msg:
                error_msg += '\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method.'
            raise RuntimeError(f'Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}')
        if is_quantized:
            unexpected_keys = [elem for elem in unexpected_keys if 'SCB' not in elem]
            missing_keys = [elem for elem in missing_keys if 'SCB' not in elem]
        if len(unexpected_keys) > 0:
            archs = [] if model.config.architectures is None else model.config.architectures
            warner = logger.warning if model.__class__.__name__ in archs else logger.info
            warner(f'Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).\n- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).')
        else:
            logger.info(f'All model checkpoint weights were used when initializing {model.__class__.__name__}.\n')
        if len(missing_keys) > 0:
            logger.warning(f'Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.')
        elif len(mismatched_keys) == 0:
            logger.info(f'All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint was trained on, you can already use {model.__class__.__name__} for predictions without further training.')
        if len(mismatched_keys) > 0:
            mismatched_warning = '\n'.join([f'- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated' for (key, shape1, shape2) in mismatched_keys])
            logger.warning(f'Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} and are newly initialized because the shapes did not match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.')
        return (model, missing_keys, unexpected_keys, mismatched_keys, offload_index, error_msgs)

    def retrieve_modules_from_names(self, names, add_prefix=False, remove_prefix=False):
        if False:
            i = 10
            return i + 15
        module_keys = {'.'.join(key.split('.')[:-1]) for key in names}
        module_keys = module_keys.union({'.'.join(key.split('.')[:-2]) for key in names if len(key) > 0 and key[-1].isdigit()})
        retrieved_modules = []
        for (name, module) in self.named_modules():
            if remove_prefix:
                _prefix = f'{self.base_model_prefix}.'
                name = name[len(_prefix):] if name.startswith(_prefix) else name
            elif add_prefix:
                name = '.'.join([self.base_model_prefix, name]) if len(name) > 0 else self.base_model_prefix
            if name in module_keys:
                retrieved_modules.append(module)
        return retrieved_modules

    @staticmethod
    def _load_pretrained_model_low_mem(model, loaded_state_dict_keys, resolved_archive_file, start_prefix=''):
        if False:
            i = 10
            return i + 15
        "\n        This is an experimental function that loads the model using ~1.x model size CPU memory\n\n        Before you call it do:\n\n        1. save which state_dict keys are available\n        2. drop state_dict before model is created, since the latter takes 1x model size memory\n\n        Here then we continue:\n\n        3. switch to the meta device all params/buffers that are going to be replaced from the loaded state_dict\n        4. load state_dict 2nd time\n        5. replace the params/buffers from the state_dict\n\n        Currently, it doesn't handle missing_keys, unexpected_keys, mismatched_keys. It can't handle deepspeed.\n        "
        _move_model_to_meta(model, loaded_state_dict_keys, start_prefix)
        state_dict = load_state_dict(resolved_archive_file)
        error_msgs = _load_state_dict_into_meta_model(model, state_dict, loaded_state_dict_keys, start_prefix)
        return error_msgs

    @classmethod
    def register_for_auto_class(cls, auto_class='AutoModel'):
        if False:
            print('Hello World!')
        '\n        Register this class with a given auto class. This should only be used for custom models as the ones in the\n        library are already mapped with an auto class.\n\n        <Tip warning={true}>\n\n        This API is experimental and may have some slight breaking changes in the next releases.\n\n        </Tip>\n\n        Args:\n            auto_class (`str` or `type`, *optional*, defaults to `"AutoModel"`):\n                The auto class to register this new model with.\n        '
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__
        import transformers.models.auto as auto_module
        if not hasattr(auto_module, auto_class):
            raise ValueError(f'{auto_class} is not a valid auto class.')
        cls._auto_class = auto_class

    def to_bettertransformer(self) -> 'PreTrainedModel':
        if False:
            i = 10
            return i + 15
        "\n        Converts the model to use [PyTorch's native attention\n        implementation](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html), integrated to\n        Transformers through [Optimum library](https://huggingface.co/docs/optimum/bettertransformer/overview). Only a\n        subset of all Transformers models are supported.\n\n        PyTorch's attention fastpath allows to speed up inference through kernel fusions and the use of [nested\n        tensors](https://pytorch.org/docs/stable/nested.html). Detailed benchmarks can be found in [this blog\n        post](https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2).\n\n        Returns:\n            [`PreTrainedModel`]: The model converted to BetterTransformer.\n        "
        if not is_optimum_available():
            raise ImportError('The package `optimum` is required to use Better Transformer.')
        from optimum.version import __version__ as optimum_version
        if version.parse(optimum_version) < version.parse('1.7.0'):
            raise ImportError(f'Please install optimum>=1.7.0 to use Better Transformer. The version {optimum_version} was found.')
        from optimum.bettertransformer import BetterTransformer
        return BetterTransformer.transform(self)

    def reverse_bettertransformer(self):
        if False:
            while True:
                i = 10
        '\n        Reverts the transformation from [`~PreTrainedModel.to_bettertransformer`] so that the original modeling is\n        used, for example in order to save the model.\n\n        Returns:\n            [`PreTrainedModel`]: The model converted back to the original modeling.\n        '
        if not is_optimum_available():
            raise ImportError('The package `optimum` is required to use Better Transformer.')
        from optimum.version import __version__ as optimum_version
        if version.parse(optimum_version) < version.parse('1.7.0'):
            raise ImportError(f'Please install optimum>=1.7.0 to use Better Transformer. The version {optimum_version} was found.')
        from optimum.bettertransformer import BetterTransformer
        return BetterTransformer.reverse(self)

    def warn_if_padding_and_no_attention_mask(self, input_ids, attention_mask):
        if False:
            return 10
        '\n        Shows a one-time warning if the input_ids appear to contain padding and no attention mask was given.\n        '
        if is_torch_fx_proxy(input_ids) or torch.jit.is_tracing() or is_torchdynamo_compiling():
            return
        if attention_mask is not None or self.config.pad_token_id is None:
            return
        if self.config.pad_token_id in input_ids[:, [-1, 0]]:
            warn_string = 'We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See https://huggingface.co/docs/transformers/troubleshooting#incorrect-output-when-padding-tokens-arent-masked.'
            if self.config.bos_token_id is not None and self.config.bos_token_id == self.config.pad_token_id or (self.config.eos_token_id is not None and self.config.eos_token_id == self.config.pad_token_id) or (self.config.sep_token_id is not None and self.config.sep_token_id == self.config.pad_token_id):
                warn_string += f'\nYou may ignore this warning if your `pad_token_id` ({self.config.pad_token_id}) is identical to the `bos_token_id` ({self.config.bos_token_id}), `eos_token_id` ({self.config.eos_token_id}), or the `sep_token_id` ({self.config.sep_token_id}), and your input is not padded.'
            logger.warning_once(warn_string)
PreTrainedModel.push_to_hub = copy_func(PreTrainedModel.push_to_hub)
if PreTrainedModel.push_to_hub.__doc__ is not None:
    PreTrainedModel.push_to_hub.__doc__ = PreTrainedModel.push_to_hub.__doc__.format(object='model', object_class='AutoModel', object_files='model file')

class PoolerStartLogits(nn.Module):
    """
    Compute SQuAD start logits from sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model.
    """

    def __init__(self, config: PretrainedConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states: torch.FloatTensor, p_mask: Optional[torch.FloatTensor]=None) -> torch.FloatTensor:
        if False:
            while True:
                i = 10
        '\n        Args:\n            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):\n                The final hidden states of the model.\n            p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):\n                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token\n                should be masked.\n\n        Returns:\n            `torch.FloatTensor`: The start logits for SQuAD.\n        '
        x = self.dense(hidden_states).squeeze(-1)
        if p_mask is not None:
            if get_parameter_dtype(self) == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e+30 * p_mask
        return x

class PoolerEndLogits(nn.Module):
    """
    Compute SQuAD end logits from sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model and the `layer_norm_eps`
            to use.
    """

    def __init__(self, config: PretrainedConfig):
        if False:
            print('Hello World!')
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states: torch.FloatTensor, start_states: Optional[torch.FloatTensor]=None, start_positions: Optional[torch.LongTensor]=None, p_mask: Optional[torch.FloatTensor]=None) -> torch.FloatTensor:
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):\n                The final hidden states of the model.\n            start_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):\n                The hidden states of the first tokens for the labeled span.\n            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n                The position of the first token for the labeled span.\n            p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):\n                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token\n                should be masked.\n\n        <Tip>\n\n        One of `start_states` or `start_positions` should be not `None`. If both are set, `start_positions` overrides\n        `start_states`.\n\n        </Tip>\n\n        Returns:\n            `torch.FloatTensor`: The end logits for SQuAD.\n        '
        assert start_states is not None or start_positions is not None, 'One of start_states, start_positions should be not None'
        if start_positions is not None:
            (slen, hsz) = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)
            start_states = hidden_states.gather(-2, start_positions)
            start_states = start_states.expand(-1, slen, -1)
        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)
        if p_mask is not None:
            if get_parameter_dtype(self) == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e+30 * p_mask
        return x

class PoolerAnswerClass(nn.Module):
    """
    Compute SQuAD 2.0 answer class from classification and start tokens hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model.
    """

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.dense_1 = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.FloatTensor, start_states: Optional[torch.FloatTensor]=None, start_positions: Optional[torch.LongTensor]=None, cls_index: Optional[torch.LongTensor]=None) -> torch.FloatTensor:
        if False:
            print('Hello World!')
        '\n        Args:\n            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):\n                The final hidden states of the model.\n            start_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):\n                The hidden states of the first tokens for the labeled span.\n            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n                The position of the first token for the labeled span.\n            cls_index (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n                Position of the CLS token for each sentence in the batch. If `None`, takes the last token.\n\n        <Tip>\n\n        One of `start_states` or `start_positions` should be not `None`. If both are set, `start_positions` overrides\n        `start_states`.\n\n        </Tip>\n\n        Returns:\n            `torch.FloatTensor`: The SQuAD 2.0 answer class.\n        '
        hsz = hidden_states.shape[-1]
        assert start_states is not None or start_positions is not None, 'One of start_states, start_positions should be not None'
        if start_positions is not None:
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)
            start_states = hidden_states.gather(-2, start_positions).squeeze(-2)
        if cls_index is not None:
            cls_index = cls_index[:, None, None].expand(-1, -1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2)
        else:
            cls_token_state = hidden_states[:, -1, :]
        x = self.dense_0(torch.cat([start_states, cls_token_state], dim=-1))
        x = self.activation(x)
        x = self.dense_1(x).squeeze(-1)
        return x

@dataclass
class SquadHeadOutput(ModelOutput):
    """
    Base class for outputs of question answering models using a [`~modeling_utils.SQuADHead`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided):
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification
            losses.
        start_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        start_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Indices for the top config.start_n_top start token possibilities (beam-search).
        end_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities
            (beam-search).
        end_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
        cls_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the `is_impossible` label of the answers.

    """
    loss: Optional[torch.FloatTensor] = None
    start_top_log_probs: Optional[torch.FloatTensor] = None
    start_top_index: Optional[torch.LongTensor] = None
    end_top_log_probs: Optional[torch.FloatTensor] = None
    end_top_index: Optional[torch.LongTensor] = None
    cls_logits: Optional[torch.FloatTensor] = None

class SQuADHead(nn.Module):
    """
    A SQuAD head inspired by XLNet.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model and the `layer_norm_eps`
            to use.
    """

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

    @replace_return_docstrings(output_type=SquadHeadOutput, config_class=PretrainedConfig)
    def forward(self, hidden_states: torch.FloatTensor, start_positions: Optional[torch.LongTensor]=None, end_positions: Optional[torch.LongTensor]=None, cls_index: Optional[torch.LongTensor]=None, is_impossible: Optional[torch.LongTensor]=None, p_mask: Optional[torch.FloatTensor]=None, return_dict: bool=False) -> Union[SquadHeadOutput, Tuple[torch.FloatTensor]]:
        if False:
            while True:
                i = 10
        '\n        Args:\n            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):\n                Final hidden states of the model on the sequence tokens.\n            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n                Positions of the first token for the labeled span.\n            end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n                Positions of the last token for the labeled span.\n            cls_index (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n                Position of the CLS token for each sentence in the batch. If `None`, takes the last token.\n            is_impossible (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n                Whether the question has a possible answer in the paragraph or not.\n            p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):\n                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token\n                should be masked.\n            return_dict (`bool`, *optional*, defaults to `False`):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n\n        Returns:\n        '
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)
        if start_positions is not None and end_positions is not None:
            for x in (start_positions, end_positions, cls_index, is_impossible):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)
            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            if cls_index is not None and is_impossible is not None:
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)
                total_loss += cls_loss * 0.5
            return SquadHeadOutput(loss=total_loss) if return_dict else (total_loss,)
        else:
            (bsz, slen, hsz) = hidden_states.size()
            start_log_probs = nn.functional.softmax(start_logits, dim=-1)
            (start_top_log_probs, start_top_index) = torch.topk(start_log_probs, self.start_n_top, dim=-1)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)
            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(start_states)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = nn.functional.softmax(end_logits, dim=1)
            (end_top_log_probs, end_top_index) = torch.topk(end_log_probs, self.end_n_top, dim=1)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)
            start_states = torch.einsum('blh,bl->bh', hidden_states, start_log_probs)
            cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)
            if not return_dict:
                return (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits)
            else:
                return SquadHeadOutput(start_top_log_probs=start_top_log_probs, start_top_index=start_top_index, end_top_log_probs=end_top_log_probs, end_top_index=end_top_index, cls_logits=cls_logits)

class SequenceSummary(nn.Module):
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
    """

    def __init__(self, config: PretrainedConfig):
        if False:
            return 10
        super().__init__()
        self.summary_type = getattr(config, 'summary_type', 'last')
        if self.summary_type == 'attn':
            raise NotImplementedError
        self.summary = Identity()
        if hasattr(config, 'summary_use_proj') and config.summary_use_proj:
            if hasattr(config, 'summary_proj_to_labels') and config.summary_proj_to_labels and (config.num_labels > 0):
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)
        activation_string = getattr(config, 'summary_activation', None)
        self.activation: Callable = get_activation(activation_string) if activation_string else Identity()
        self.first_dropout = Identity()
        if hasattr(config, 'summary_first_dropout') and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)
        self.last_dropout = Identity()
        if hasattr(config, 'summary_last_dropout') and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def forward(self, hidden_states: torch.FloatTensor, cls_index: Optional[torch.LongTensor]=None) -> torch.FloatTensor:
        if False:
            print('Hello World!')
        '\n        Compute a single vector summary of a sequence hidden states.\n\n        Args:\n            hidden_states (`torch.FloatTensor` of shape `[batch_size, seq_len, hidden_size]`):\n                The hidden states of the last layer.\n            cls_index (`torch.LongTensor` of shape `[batch_size]` or `[batch_size, ...]` where ... are optional leading dimensions of `hidden_states`, *optional*):\n                Used if `summary_type == "cls_index"` and takes the last token of the sequence as classification token.\n\n        Returns:\n            `torch.FloatTensor`: The summary of the sequence hidden states.\n        '
        if self.summary_type == 'last':
            output = hidden_states[:, -1]
        elif self.summary_type == 'first':
            output = hidden_states[:, 0]
        elif self.summary_type == 'mean':
            output = hidden_states.mean(dim=1)
        elif self.summary_type == 'cls_index':
            if cls_index is None:
                cls_index = torch.full_like(hidden_states[..., :1, :], hidden_states.shape[-2] - 1, dtype=torch.long)
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
            output = hidden_states.gather(-2, cls_index).squeeze(-2)
        elif self.summary_type == 'attn':
            raise NotImplementedError
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)
        return output

def unwrap_model(model: nn.Module) -> nn.Module:
    if False:
        while True:
            i = 10
    '\n    Recursively unwraps a model from potential containers (as used in distributed training).\n\n    Args:\n        model (`torch.nn.Module`): The model to unwrap.\n    '
    if hasattr(model, 'module'):
        return unwrap_model(model.module)
    else:
        return model

def expand_device_map(device_map, param_names, start_prefix):
    if False:
        print('Hello World!')
    '\n    Expand a device map to return the correspondance parameter name to device.\n    '
    new_device_map = {}
    param_names = [p[len(start_prefix):] for p in param_names if p.startswith(start_prefix)]
    for (module, device) in device_map.items():
        new_device_map.update({p: device for p in param_names if p == module or p.startswith(f'{module}.') or module == ''})
    return new_device_map

def get_disk_only_shard_files(device_map, sharded_metadata, start_prefix):
    if False:
        print('Hello World!')
    '\n    Returns the list of shard files containing only weights offloaded to disk.\n    '
    weight_map = {p[len(start_prefix):]: v for (p, v) in sharded_metadata['weight_map'].items() if p.startswith(start_prefix)}
    files_content = collections.defaultdict(list)
    for (weight_name, filename) in weight_map.items():
        while len(weight_name) > 0 and weight_name not in device_map:
            weight_name = '.'.join(weight_name.split('.')[:-1])
        files_content[filename].append(device_map[weight_name])
    return [fname for (fname, devices) in files_content.items() if set(devices) == {'disk'}]