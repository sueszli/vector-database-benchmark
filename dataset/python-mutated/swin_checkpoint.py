import os.path as osp
import pkgutil
from collections import OrderedDict
from importlib import import_module
import torch
import torchvision
from torch.nn import functional as F

def load_state_dict(module, state_dict, strict=False, logger=None):
    if False:
        for i in range(10):
            print('nop')
    "Load state_dict to a module.\n\n    This method is modified from :meth:`torch.nn.Module.load_state_dict`.\n    Default value for ``strict`` is set to ``False`` and the message for\n    param mismatch will be shown even if strict is False.\n    Args:\n        module (Module): Module that receives the state_dict.\n        state_dict (OrderedDict): Weights.\n        strict (bool): whether to strictly enforce that the keys\n            in :attr:`state_dict` match the keys returned by this module's\n            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.\n        logger (:obj:`logging.Logger`, optional): Logger to log the error\n            message. If not specified, print function will be used.\n    "
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        if False:
            while True:
                i = 10
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True, all_missing_keys, unexpected_keys, err_msg)
        for (name, child) in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
    load(module)
    load = None
    missing_keys = [key for key in all_missing_keys if 'num_batches_tracked' not in key]
    if unexpected_keys:
        err_msg.append(f"unexpected key in source state_dict: {', '.join(unexpected_keys)}\n")
    if missing_keys:
        err_msg.append(f"missing keys in source state_dict: {', '.join(missing_keys)}\n")
    if len(err_msg) > 0:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)

def get_torchvision_models():
    if False:
        while True:
            i = 10
    model_urls = dict()
    for (_, name, ispkg) in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module(f'torchvision.models.{name}')
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    return model_urls

def _process_mmcls_checkpoint(checkpoint):
    if False:
        for i in range(10):
            print('nop')
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for (k, v) in state_dict.items():
        if k.startswith('backbone.'):
            new_state_dict[k[9:]] = v
    new_checkpoint = dict(state_dict=new_state_dict)
    return new_checkpoint

def _load_checkpoint(filename, map_location=None):
    if False:
        print('Hello World!')
    'Load checkpoint from somewhere (modelzoo, file, url).\n\n    Args:\n        filename (str): Accept local filepath, URL, ``torchvision://xxx``,\n            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for\n            details.\n        map_location (str | None): Same as :func:`torch.load`. Default: None.\n    Returns:\n        dict | OrderedDict: The loaded checkpoint. It can be either an\n            OrderedDict storing model weights or a dict containing other\n            information, which depends on the checkpoint.\n    '
    if not osp.isfile(filename):
        raise IOError(f'{filename} is not a checkpoint file')
    checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint

def load_checkpoint(model, filename, map_location='cpu', strict=False, logger=None):
    if False:
        while True:
            i = 10
    'Load checkpoint from a file or URI.\n\n    Args:\n        model (Module): Module to load checkpoint.\n        filename (str): Accept local filepath, URL, ``torchvision://xxx``,\n            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for\n            details.\n        map_location (str): Same as :func:`torch.load`.\n        strict (bool): Whether to allow different params for the model and\n            checkpoint.\n        logger (:mod:`logging.Logger` or None): The logger for error message.\n    Returns:\n        dict or OrderedDict: The loaded checkpoint.\n    '
    checkpoint = _load_checkpoint(filename, map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for (k, v) in state_dict.items()}
    if state_dict.get('absolute_pos_embed') is not None:
        absolute_pos_embed = state_dict['absolute_pos_embed']
        (N1, L, C1) = absolute_pos_embed.size()
        (N2, C2, H, W) = model.absolute_pos_embed.size()
        if N1 != N2 or C1 != C2 or L != H * W:
            logger.warning('Error in loading absolute_pos_embed, pass')
        else:
            state_dict['absolute_pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2)
    relative_position_bias_table_keys = [k for k in state_dict.keys() if 'relative_position_bias_table' in k]
    for table_key in relative_position_bias_table_keys:
        table_pretrained = state_dict[table_key]
        table_current = model.state_dict()[table_key]
        (L1, nH1) = table_pretrained.size()
        (L2, nH2) = table_current.size()
        if nH1 != nH2:
            logger.warning(f'Error in loading {table_key}, pass')
        elif L1 != L2:
            S1 = int(L1 ** 0.5)
            S2 = int(L2 ** 0.5)
            table_pretrained_resized = F.interpolate(table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2), mode='bicubic')
            state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0)
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint

def weights_to_cpu(state_dict):
    if False:
        for i in range(10):
            print('nop')
    'Copy a model state_dict to cpu.\n\n    Args:\n        state_dict (OrderedDict): Model weights on GPU.\n    Returns:\n        OrderedDict: Model weights on GPU.\n    '
    state_dict_cpu = OrderedDict()
    for (key, val) in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu

def _save_to_state_dict(module, destination, prefix, keep_vars):
    if False:
        i = 10
        return i + 15
    'Saves module state to `destination` dictionary.\n\n    This method is modified from :meth:`torch.nn.Module._save_to_state_dict`.\n    Args:\n        module (nn.Module): The module to generate state_dict.\n        destination (dict): A dict where state will be stored.\n        prefix (str): The prefix for parameters and buffers used in this\n            module.\n    '
    for (name, param) in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for (name, buf) in module._buffers.items():
        if buf is not None:
            destination[prefix + name] = buf if keep_vars else buf.detach()

def get_state_dict(module, destination=None, prefix='', keep_vars=False):
    if False:
        for i in range(10):
            print('nop')
    'Returns a dictionary containing a whole state of the module.\n\n    Both parameters and persistent buffers (e.g. running averages) are\n    included. Keys are corresponding parameter and buffer names.\n    This method is modified from :meth:`torch.nn.Module.state_dict` to\n    recursively check parallel module in case that the model has a complicated\n    structure, e.g., nn.Module(nn.Module(DDP)).\n    Args:\n        module (nn.Module): The module to generate state_dict.\n        destination (OrderedDict): Returned dict for the state of the\n            module.\n        prefix (str): Prefix of the key.\n        keep_vars (bool): Whether to keep the variable property of the\n            parameters. Default: False.\n    Returns:\n        dict: A dictionary containing a whole state of the module.\n    '
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(version=module._version)
    _save_to_state_dict(module, destination, prefix, keep_vars)
    for (name, child) in module._modules.items():
        if child is not None:
            get_state_dict(child, destination, prefix + name + '.', keep_vars=keep_vars)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination