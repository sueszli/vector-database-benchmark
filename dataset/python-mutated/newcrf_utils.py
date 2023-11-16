import os
import os.path as osp
import pkgutil
import warnings
from collections import OrderedDict
from importlib import import_module
import torch
import torch.nn as nn
import torchvision
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils import model_zoo
TORCH_VERSION = torch.__version__

def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    if False:
        i = 10
        return i + 15
    if warning:
        if size is not None and align_corners:
            (input_h, input_w) = tuple((int(x) for x in input.shape[2:]))
            (output_h, output_w) = tuple((int(x) for x in size))
            if output_h > input_h or output_w > output_h:
                if (output_h > 1 and output_w > 1 and (input_h > 1) and (input_w > 1)) and (output_h - 1) % (input_h - 1) and (output_w - 1) % (input_w - 1):
                    warnings.warn(f'When align_corners={align_corners}, the output would more aligned if input size {(input_h, input_w)} is `x+1` and out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple((int(x) for x in size))
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def normal_init(module, mean=0, std=1, bias=0):
    if False:
        print('Hello World!')
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def is_module_wrapper(module):
    if False:
        return 10
    module_wrappers = (DataParallel, DistributedDataParallel)
    return isinstance(module, module_wrappers)

def get_dist_info():
    if False:
        for i in range(10):
            print('nop')
    if TORCH_VERSION < '1.0':
        initialized = dist._initialized
    elif dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return (rank, world_size)

def load_state_dict(module, state_dict, strict=False, logger=None):
    if False:
        for i in range(10):
            print('nop')
    "Load state_dict to a module.\n\n    This method is modified from :meth:`torch.nn.Module.load_state_dict`.\n    Default value for ``strict`` is set to ``False`` and the message for\n    param mismatch will be shown even if strict is False.\n\n    Args:\n        module (Module): Module that receives the state_dict.\n        state_dict (OrderedDict): Weights.\n        strict (bool): whether to strictly enforce that the keys\n            in :attr:`state_dict` match the keys returned by this module's\n            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.\n        logger (:obj:`logging.Logger`, optional): Logger to log the error\n            message. If not specified, print function will be used.\n    "
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        if False:
            return 10
        if is_module_wrapper(module):
            module = module.module
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
    (rank, _) = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)

def load_url_dist(url, model_dir=None):
    if False:
        return 10
    'In distributed setting, this function only download checkpoint at local\n    rank 0.'
    (rank, world_size) = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    return checkpoint

def get_torchvision_models():
    if False:
        i = 10
        return i + 15
    model_urls = dict()
    for (_, name, ispkg) in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module(f'torchvision.models.{name}')
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    return model_urls

def _load_checkpoint(filename, map_location=None):
    if False:
        print('Hello World!')
    'Load checkpoint from somewhere (modelzoo, file, url).\n\n    Args:\n        filename (str): Accept local filepath, URL, ``torchvision://xxx``,\n            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for\n            details.\n        map_location (str | None): Same as :func:`torch.load`. Default: None.\n\n    Returns:\n        dict | OrderedDict: The loaded checkpoint. It can be either an\n            OrderedDict storing model weights or a dict containing other\n            information, which depends on the checkpoint.\n    '
    if filename.startswith('modelzoo://'):
        warnings.warn('The URL scheme of "modelzoo://" is deprecated, please use "torchvision://" instead')
        model_urls = get_torchvision_models()
        model_name = filename[11:]
        checkpoint = load_url_dist(model_urls[model_name])
    else:
        if not osp.isfile(filename):
            raise IOError(f'{filename} is not a checkpoint file')
        checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint

def load_checkpoint(model, filename, map_location='cpu', strict=False, logger=None):
    if False:
        i = 10
        return i + 15
    'Load checkpoint from a file or URI.\n\n    Args:\n        model (Module): Module to load checkpoint.\n        filename (str): Accept local filepath, URL, ``torchvision://xxx``,\n            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for\n            details.\n        map_location (str): Same as :func:`torch.load`.\n        strict (bool): Whether to allow different params for the model and\n            checkpoint.\n        logger (:mod:`logging.Logger` or None): The logger for error message.\n\n    Returns:\n        dict or OrderedDict: The loaded checkpoint.\n    '
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
    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for (k, v) in state_dict.items() if k.startswith('encoder.')}
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