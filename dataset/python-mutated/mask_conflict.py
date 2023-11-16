import logging
from typing import Dict
import numpy as np
import torch
import torch.fx
from .dependency import build_channel_dependency, build_group_dependency, build_weight_sharing_dependency
_logger = logging.getLogger(__name__)

def fix_group_mask_conflict(graph_module: torch.fx.GraphModule, masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if False:
        while True:
            i = 10
    '\n    Fix the mask conflict between group and channel.\n    This function will modify the masks in-place.\n\n    Parameters\n    ----------\n    graph_module\n        The graph module.\n    masks\n        The masks of all modules.\n\n    Returns\n    -------\n    masks\n        The fixed masks.\n    '
    group_dependency = build_group_dependency(graph_module)
    for (node, (max_group, min_group)) in group_dependency.items():
        layername = node.target
        if layername not in masks or 'weight' not in masks[layername]:
            continue
        w_mask = masks[layername]['weight']
        shape = w_mask.shape
        count = np.prod(shape[1:])
        all_ones = (w_mask.flatten(1).sum(-1) == count).nonzero().squeeze(1).tolist()
        all_zeros = (w_mask.flatten(1).sum(-1) == 0).nonzero().squeeze(1).tolist()
        if len(all_ones) + len(all_zeros) < w_mask.size(0):
            _logger.info('Layers %s using fine-grained pruning', layername)
            continue
        assert shape[0] % max_group == 0
        step = shape[0] / max_group
        group_masked = []
        for i in range(max_group):
            _start = step * i
            _end = step * (i + 1)
            _tmp_list = list(filter(lambda x: _start <= x and x < _end, all_zeros))
            group_masked.append(_tmp_list)
        mini_masked = min([len(x) for x in group_masked])
        need_unmask = set()
        for gm in group_masked:
            for i in range(mini_masked, len(gm)):
                pos = gm[i]
                need_unmask.add(pos)
        step = shape[0] / min_group
        for i in range(min_group):
            _start = step * i
            _end = step * (i + 1)
            _tmp_list = list(filter(lambda x: _start <= x and x < _end, all_zeros))
            if len(_tmp_list) == step:
                for pos in _tmp_list:
                    if pos in need_unmask:
                        need_unmask.remove(pos)
        for pos in need_unmask:
            masks[layername]['weight'][pos] = torch.ones(shape[1:])
            if hasattr(masks[layername], 'bias'):
                masks[layername]['bias'][pos] = 1
    return masks

def fix_weight_sharing_mask_conflict(graph_module: torch.fx.GraphModule, masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if False:
        print('Hello World!')
    '\n    Fix the mask conflict produced by weight sharing.\n    This function will modify the masks in-place.\n\n    Parameters\n    ----------\n    graph_module\n        The graph module.\n    masks\n        The masks of all modules.\n\n    Returns\n    -------\n    masks\n        The fixed masks.\n    '
    dependency_sets = build_weight_sharing_dependency(graph_module)
    prune_axis = detect_mask_prune_dim(graph_module, masks)
    sum_idx = (1, 2, 3) if prune_axis == 0 else (0, 2, 3)
    (_, _tmp_tensor) = list(masks.items())[0]
    device = list(_tmp_tensor.values())[0].device
    for d_set in dependency_sets:
        if len(d_set) <= 1:
            continue
        channel_masks = []
        fine_grained = False
        for name in d_set:
            if name in masks and 'weight' in masks[name]:
                sub_module = graph_module.get_submodule(name)
                assert sub_module is not None
                mask = masks[name]['weight']
                if isinstance(sub_module, torch.nn.Conv2d):
                    channel_mask = (mask.abs().sum(sum_idx) != 0).int()
                    if prune_axis == 1:
                        channel_mask = channel_mask.repeat(sub_module.groups)
                    channel_masks.append(channel_mask)
                    if (channel_mask.sum() * (mask.numel() / mask.shape[prune_axis])).item() != (mask > 0).sum().item():
                        fine_grained = True
                elif isinstance(sub_module, torch.nn.Linear):
                    if prune_axis == 1:
                        channel_masks.append((mask.abs().sum(0) != 0).int())
                    else:
                        channel_masks.append((mask.abs().sum(1) != 0).int())
                elif isinstance(sub_module, torch.nn.Embedding):
                    if prune_axis == 0:
                        channel_masks.append((mask.abs().sum(0) != 0).int())
                elif isinstance(sub_module, torch.nn.ConvTranspose2d):
                    tmp_sum_idx = (0, 2, 3) if prune_axis == 0 else (1, 2, 3)
                    channel_mask = (mask.abs().sum(tmp_sum_idx) != 0).int()
                    if prune_axis == 0:
                        channel_mask = channel_mask.repeat(sub_module.groups)
                    channel_masks.append(channel_mask)
                    if (channel_mask.sum() * (mask.numel() / mask.shape[1 - prune_axis])).item() != (mask > 0).sum().item():
                        fine_grained = True
                else:
                    raise RuntimeError(f'unsupported module type: {type(sub_module).__name__}')
            else:
                channel_masks.append(None)
        if fine_grained:
            _logger.info('Fine-grianed mask detected')
        if all((x is None for x in channel_masks)):
            continue
        num_channels_list = [len(x) for x in channel_masks if x is not None]
        assert len(set(num_channels_list)) == 1
        num_channels = num_channels_list[0]
        for (i, dim_mask) in enumerate(channel_masks):
            if dim_mask is None:
                channel_masks[i] = torch.ones(num_channels).int().to(device)
        merged_channel_mask = channel_masks[0].clone()
        for i in range(1, len(channel_masks)):
            merged_channel_mask = (merged_channel_mask + channel_masks[i] != 0).int()
        merged_index = torch.nonzero(merged_channel_mask, as_tuple=True)[0]
        for name in d_set:
            if name not in masks or 'weight' not in masks[name]:
                assert all(merged_channel_mask)
                continue
            orig_mask = masks[name]['weight']
            sub_module = graph_module.get_submodule(name)
            new_mask = torch.zeros_like(orig_mask)
            if isinstance(sub_module, torch.nn.Conv2d):
                if prune_axis == 0:
                    new_mask[merged_index, :, :, :] = 1.0
                else:
                    new_mask[:, torch.nonzero(merged_channel_mask[:new_mask.shape[1]], as_tuple=True)[0], :, :] = 1.0
            elif isinstance(sub_module, torch.nn.Linear):
                if prune_axis == 0:
                    new_mask[merged_index, :] = 1.0
                elif prune_axis == 1:
                    new_mask[:, merged_index] = 1.0
            elif isinstance(sub_module, torch.nn.Embedding):
                if prune_axis == 0:
                    new_mask[:, merged_index] = 1.0
            else:
                raise RuntimeError(f'unsupported module type: {type(sub_module).__name__}')
            masks[name]['weight'] = new_mask
            if 'bias' in masks[name] and masks[name]['bias'] is not None:
                if prune_axis == 0:
                    masks[name]['bias'] = merged_channel_mask.type_as(masks[name]['bias'])
    return masks

def fix_channel_mask_conflict(graph_module: torch.fx.GraphModule, masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if False:
        print('Hello World!')
    '\n    Fix the mask conflict between channel and group.\n    This function will modify the masks in-place.\n\n    Parameters\n    ----------\n    graph_module\n        The graph module.\n    masks\n        The masks of all modules.\n\n    Returns\n    -------\n    masks\n        The fixed masks.\n    '
    prune_axis = detect_mask_prune_dim(graph_module, masks)
    prune_type = detect_channel_prune_type(graph_module, masks)
    print(prune_axis, prune_type)
    dependency_sets = build_channel_dependency(graph_module, prune_axis=prune_axis, prune_type=prune_type)
    sum_idx = (1, 2, 3) if prune_axis == 0 else (0, 2, 3)
    (_, _tmp_tensor) = list(masks.items())[0]
    device = list(_tmp_tensor.values())[0].device
    for d_set in dependency_sets:
        if len(d_set) <= 1:
            continue
        channel_masks = []
        fine_grained = False
        for node in d_set:
            name = node.target
            if name in masks and 'weight' in masks[name]:
                sub_module = graph_module.get_submodule(name)
                assert sub_module is not None
                mask = masks[name]['weight']
                if isinstance(sub_module, torch.nn.Conv2d):
                    channel_mask = (mask.abs().sum(sum_idx) != 0).int()
                    if prune_axis == 1:
                        channel_mask = channel_mask.repeat(sub_module.groups)
                    channel_masks.append(channel_mask)
                    if (channel_mask.sum() * (mask.numel() / mask.shape[prune_axis])).item() != (mask > 0).sum().item():
                        fine_grained = True
                elif isinstance(sub_module, torch.nn.Linear):
                    if prune_axis == 1:
                        channel_masks.append((mask.abs().sum(0) != 0).int())
                    else:
                        channel_masks.append((mask.abs().sum(1) != 0).int())
                elif isinstance(sub_module, torch.nn.Embedding):
                    if prune_axis == 0:
                        channel_masks.append((mask.abs().sum(0) != 0).int())
                elif isinstance(sub_module, torch.nn.BatchNorm2d):
                    channel_masks.append(mask.int())
                elif isinstance(sub_module, torch.nn.ConvTranspose2d):
                    tmp_sum_idx = (0, 2, 3) if prune_axis == 0 else (1, 2, 3)
                    channel_mask = (mask.abs().sum(tmp_sum_idx) != 0).int()
                    if prune_axis == 0:
                        channel_mask = channel_mask.repeat(sub_module.groups)
                    channel_masks.append(channel_mask)
                    if (channel_mask.sum() * (mask.numel() / mask.shape[1 - prune_axis])).item() != (mask > 0).sum().item():
                        fine_grained = True
                else:
                    raise RuntimeError(f'unsupported module type: {type(sub_module).__name__}')
            else:
                channel_masks.append(None)
        if fine_grained:
            _logger.info('Fine-grained mask detected')
        if all((x is None for x in channel_masks)):
            continue
        num_channels_list = [len(x) for x in channel_masks if x is not None]
        assert len(set(num_channels_list)) == 1
        num_channels = num_channels_list[0]
        for (i, dim_mask) in enumerate(channel_masks):
            if dim_mask is None:
                channel_masks[i] = torch.ones(num_channels).int().to(device)
        merged_channel_mask = channel_masks[0].clone()
        for i in range(1, len(channel_masks)):
            merged_channel_mask = (merged_channel_mask + channel_masks[i] != 0).int()
        merged_index = torch.nonzero(merged_channel_mask, as_tuple=True)[0]
        for node in d_set:
            name = node.target
            if name not in masks or 'weight' not in masks[name]:
                assert all(merged_channel_mask)
                continue
            orig_mask = masks[name]['weight']
            sub_module = graph_module.get_submodule(name)
            new_mask = torch.zeros_like(orig_mask)
            if isinstance(sub_module, torch.nn.Conv2d):
                if prune_axis == 0:
                    new_mask[merged_index, :, :, :] = 1.0
                else:
                    new_mask[:, torch.nonzero(merged_channel_mask[:new_mask.shape[1]], as_tuple=True)[0], :, :] = 1.0
            elif isinstance(sub_module, torch.nn.Linear):
                if prune_axis == 0:
                    new_mask[merged_index, :] = 1.0
                elif prune_axis == 1:
                    new_mask[:, merged_index] = 1.0
            elif isinstance(sub_module, torch.nn.Embedding):
                if prune_axis == 0:
                    new_mask[:, merged_index] = 1.0
            elif isinstance(sub_module, torch.nn.BatchNorm2d):
                new_mask = merged_channel_mask.type_as(orig_mask)
            else:
                raise RuntimeError(f'unsupported module type: {type(sub_module).__name__}')
            masks[name]['weight'] = new_mask
            if 'bias' in masks[name] and masks[name]['bias'] is not None:
                if prune_axis == 0:
                    masks[name]['bias'] = merged_channel_mask.type_as(masks[name]['bias'])
    return masks

def detect_channel_prune_type(graph_module: torch.fx.GraphModule, masks: Dict[str, Dict[str, torch.Tensor]]) -> str:
    if False:
        i = 10
        return i + 15
    '\n    User can prune a channel through two ways: 1) prune\n    the corresponding filter of the conv layer(all the\n    filter related pruner), 2) prune the BN layers that\n    followed after a conv(Slim pruner). This function find\n    the pruning type of the masks.\n\n    Parameters\n    ----------\n    graph_module: torch.fx.GraphModule\n        GraphModule object which the mask can be applied on.\n    masks: dict\n        A dict object that stores the masks.\n\n    Returns:\n    -------\n    prune_type: str\n        Could be Filter or BatchNorm\n    '
    for layer_name in masks:
        sub_module = graph_module.get_submodule(layer_name)
        if sub_module is None or not isinstance(sub_module, torch.nn.BatchNorm2d):
            return 'Filter'
    return 'BatchNorm'

def detect_mask_prune_dim(graph_module: torch.fx.GraphModule, masks: Dict[str, Dict[str, torch.Tensor]]) -> int:
    if False:
        print('Hello World!')
    '\n    Detect how the masks of convolutional layers are pruned.\n\n    Parameters\n    ----------\n    graph_module: torch.fx.GraphModule\n        GraphModule object which the mask can be applied on.\n    masks: dict\n        A dict object that stores the masks.\n    Returns:\n    -------\n        How the masks of convolutional layers are pruned, this depends on pruning algorithms, it should\n        return 1 for masks generated by AMCPruner, and returns 0 for masks generated by the rest\n        NNI builtin pruners.\n        0: filter pruning, prune filters of weights which causes channels of output feature maps are pruned.\n        1: channel pruning, prune kernels corresponding to each input channels which causes channels of\n           input feature maps are pruned.\n    '
    (dim0_preserved, dim1_preserved) = (0.0, 0.0)
    (dim0_num, dim1_num) = (0.0, 0.0)
    for layer_name in masks:
        if 'weight' not in masks[layer_name]:
            continue
        sub_module = graph_module.get_submodule(layer_name)
        if sub_module is None or not isinstance(sub_module, torch.nn.Conv2d):
            continue
        mask = masks[layer_name]['weight'].clone()
        assert (mask >= 0).sum() == mask.numel(), 'mask values should be greater than or equal to 0.'
        mask = (mask > 0).int()
        mask = mask.view(mask.shape[0], mask.shape[1], -1)
        dim0_mask = (mask.sum((1, 2)) > 0).int()
        dim1_mask = (mask.sum((0, 2)) > 0).int()
        dim0_preserved += dim0_mask.sum().item()
        dim1_preserved += dim1_mask.sum().item()
        dim0_num += len(dim0_mask)
        dim1_num += len(dim1_mask)
    if dim0_num == 0 or dim1_num == 0:
        _logger.warning('no multi-dimension masks found.')
        return 0
    (dim0_sparsity, dim1_sparsity) = (1.0 - dim0_preserved / dim0_num, 1.0 - dim1_preserved / dim1_num)
    _logger.info('dim0 sparsity: %f', dim0_sparsity)
    _logger.info('dim1 sparsity: %f', dim1_sparsity)
    if dim0_sparsity == dim1_sparsity == 0.0:
        _logger.warning('nothing masked.')
    if dim0_sparsity > 0 and dim1_sparsity > 0:
        _logger.warning('both dim0 and dim1 masks found.')
    return 0 if dim0_sparsity >= dim1_sparsity else 1