from copy import deepcopy
import math
from typing import Dict, List, Tuple
import torch
from torch import Tensor
from torch.nn import Module

def compute_sparsity_compact2origin(origin_model: Module, compact_model: Module, config_list: List[Dict]) -> List[Dict]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Compare origin model and compact model, return the sparsity of each group mentioned in config list.\n    A group means all layer mentioned in one config.\n    e.g., a linear named 'linear1' and its weight size is [100, 100] in origin model, but in compact model,\n    the layer weight size with same layer name is [100, 50],\n    then this function will return [{'op_names': 'linear1', 'total_sparsity': 0.5}].\n    "
    compact2origin_sparsity = []
    for config in config_list:
        left_weight_num = 0
        total_weight_num = 0
        for (module_name, module) in origin_model.named_modules():
            module_type = type(module).__name__
            if 'op_types' in config and module_type not in config['op_types']:
                continue
            if 'op_names' in config and module_name not in config['op_names']:
                continue
            total_weight_num += module.weight.data.numel()
        for (module_name, module) in compact_model.named_modules():
            module_type = type(module).__name__
            if 'op_types' in config and module_type not in config['op_types']:
                continue
            if 'op_names' in config and module_name not in config['op_names']:
                continue
            left_weight_num += module.weight.data.numel()
        compact2origin_sparsity.append(deepcopy(config))
        compact2origin_sparsity[-1]['total_sparsity'] = 1 - left_weight_num / total_weight_num
    return compact2origin_sparsity

def compute_sparsity_mask2compact(compact_model: Module, compact_model_masks: Dict[str, Dict[str, Tensor]], config_list: List[Dict]):
    if False:
        while True:
            i = 10
    '\n    Apply masks on compact model, return the sparsity of each group mentioned in config list.\n    A group means all layer mentioned in one config.\n    This function count all zero elements of the masks in one group,\n    then divide by the elements number of the weights in this group to compute sparsity.\n    '
    mask2compact_sparsity = []
    for config in config_list:
        left_weight_num = 0
        total_weight_num = 0
        for (module_name, module) in compact_model.named_modules():
            module_type = type(module).__name__
            if 'op_types' in config and module_type not in config['op_types']:
                continue
            if 'op_names' in config and module_name not in config['op_names']:
                continue
            module_weight_num = module.weight.data.numel()
            total_weight_num += module_weight_num
            if module_name in compact_model_masks:
                weight_mask = compact_model_masks[module_name]['weight']
                left_weight_num += len(torch.nonzero(weight_mask, as_tuple=False))
            else:
                left_weight_num += module_weight_num
        mask2compact_sparsity.append(deepcopy(config))
        mask2compact_sparsity[-1]['total_sparsity'] = 1 - left_weight_num / total_weight_num
    return mask2compact_sparsity

def compute_sparsity(origin_model: Module, compact_model: Module, compact_model_masks: Dict[str, Dict[str, Tensor]], config_list: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    if False:
        while True:
            i = 10
    '\n    This function computes how much the origin model has been compressed in the current state.\n    The current state means `compact_model` + `compact_model_masks`\n    (i.e., `compact_model_masks` applied on `compact_model`).\n    The compact model is the origin model after pruning,\n    and it may have different structure with origin_model cause of speedup.\n\n    Parameters\n    ----------\n    origin_model : torch.nn.Module\n        The original un-pruned model.\n    compact_model : torch.nn.Module\n        The model after speedup or original model.\n    compact_model_masks: Dict[str, Dict[str, Tensor]]\n        The masks applied on the compact model, if the original model have been speedup, this should be {}.\n    config_list : List[Dict]\n        The config_list used by pruning the original model.\n\n    Returns\n    -------\n    Tuple[List[Dict], List[Dict], List[Dict]]\n        (current2origin_sparsity, compact2origin_sparsity, mask2compact_sparsity).\n        current2origin_sparsity is how much the origin model has been compressed in the current state.\n        compact2origin_sparsity is the sparsity obtained by comparing the structure of origin model and compact model.\n        mask2compact_sparsity is the sparsity computed by count the zero value in the mask.\n    '
    compact2origin_sparsity = compute_sparsity_compact2origin(origin_model, compact_model, config_list)
    mask2compact_sparsity = compute_sparsity_mask2compact(compact_model, compact_model_masks, config_list)
    assert len(compact2origin_sparsity) == len(mask2compact_sparsity), 'Length mismatch.'
    current2origin_sparsity = []
    for (c2o_sparsity, m2c_sparsity, config) in zip(compact2origin_sparsity, mask2compact_sparsity, config_list):
        current2origin_sparsity.append(deepcopy(config))
        current2origin_sparsity[-1]['total_sparsity'] = 1 - (1 - c2o_sparsity['total_sparsity']) * (1 - m2c_sparsity['total_sparsity'])
    return (current2origin_sparsity, compact2origin_sparsity, mask2compact_sparsity)

def get_model_weights_numel(model: Module, config_list: List[Dict], masks: Dict[str, Dict[str, Tensor]]={}) -> Tuple[Dict[str, int], Dict[str, float]]:
    if False:
        return 10
    '\n    Count the layer weight elements number in config_list.\n    If masks is not empty, the masked weight will not be counted.\n    '
    model_weights_numel = {}
    masked_rate = {}
    for config in config_list:
        for (module_name, module) in model.named_modules():
            module_type = type(module).__name__
            if 'op_types' in config and module_type not in config['op_types']:
                continue
            if 'op_names' in config and module_name not in config['op_names']:
                continue
            if module_name in masks and isinstance(masks[module_name]['weight'], Tensor):
                weight_mask = masks[module_name]['weight']
                masked_rate[module_name] = 1 - weight_mask.sum().item() / weight_mask.numel()
                model_weights_numel[module_name] = round(weight_mask.sum().item())
            else:
                model_weights_numel[module_name] = module.weight.data.numel()
    return (model_weights_numel, masked_rate)

def get_output_batch_dims(t: Tensor, module: Module):
    if False:
        print('Hello World!')
    if isinstance(module, (torch.nn.Linear, torch.nn.Bilinear)):
        batch_nums = math.prod(t.shape[:-1])
        batch_dims = [_ for _ in range(len(t.shape[:-1]))]
    elif isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
        batch_nums = math.prod(t.shape[:-2])
        batch_dims = [_ for _ in range(len(t.shape[:-2]))]
    elif isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        batch_nums = math.prod(t.shape[:-3])
        batch_dims = [_ for _ in range(len(t.shape[:-3]))]
    elif isinstance(module, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
        batch_nums = math.prod(t.shape[:-4])
        batch_dims = [_ for _ in range(len(t.shape[:-4]))]
    else:
        raise TypeError(f'Found unsupported module type in activation based pruner: {module.__class__.__name__}')
    return (batch_dims, batch_nums)