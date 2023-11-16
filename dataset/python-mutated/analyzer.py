import time
from collections import OrderedDict
from typing import Dict, Sequence
import functools
import itertools
import numpy as np
import torch
import torch.nn as nn
from .compute_madd import compute_madd
from .compute_flops import compute_flops
from .compute_memory import compute_memory
from .stat_tree import StatTree, StatNode
from .reporter import report_format

class ModuleStats:

    def __init__(self, name) -> None:
        if False:
            i = 10
            return i + 15
        self.name = name
        self.start_time = 0.0
        self.end_time = 0.0
        self.inference_memory = 0
        self.input_shape: Sequence[int] = []
        self.output_shape: Sequence[int] = []
        self.MAdd = 0
        self.duration = 0.0
        self.Flops = 0
        self.Memory = (0, 0)
        self.parameter_quantity = 0
        self.done = False

def print_report(self, collected_nodes):
    if False:
        while True:
            i = 10
    report = report_format(self.collected_nodes)
    print(report)

def analyze(model: nn.Module, input_size, query_granularity: int):
    if False:
        i = 10
        return i + 15
    assert isinstance(model, nn.Module)
    assert isinstance(input_size, (list, tuple))
    (pre_hooks, post_hooks) = ([], [])
    stats: OrderedDict[str, ModuleStats] = OrderedDict()
    try:
        _for_leaf(model, _register_hooks, pre_hooks, post_hooks, stats)
        x = torch.rand(*input_size)
        x = x.to(next(model.parameters()).device)
        model.eval()
        model(x)
        stat_tree = _convert_leaf_modules_to_stat_tree(stats)
        return stat_tree.get_collected_stat_nodes(query_granularity)
    finally:
        for stat in stats.values():
            stat.done = True
        for hook in itertools.chain(pre_hooks, post_hooks):
            hook.remove()

def _for_leaf(model, fn, *args):
    if False:
        for i in range(10):
            print('nop')
    for (name, module) in model.named_modules():
        if len(list(module.children())) == 0:
            fn(name, module, *args)

def _register_hooks(name: str, module: nn.Module, pre_hooks, post_hooks, stats):
    if False:
        while True:
            i = 10
    assert isinstance(module, nn.Module) and len(list(module.children())) == 0
    if name in stats:
        return
    module_stats = ModuleStats(name)
    stats[name] = module_stats
    post_hook = module.register_forward_hook(functools.partial(_forward_post_hook, module_stats))
    post_hooks.append(post_hook)
    pre_hook = module.register_forward_pre_hook(functools.partial(_forward_pre_hook, module_stats))
    pre_hooks.append(pre_hook)

def _flatten(x):
    if False:
        i = 10
        return i + 15
    'Flattens the tree of tensors to flattened sequence of tensors'
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, Sequence):
        res = []
        for xi in x:
            res += _flatten(xi)
        return res
    return []

def _forward_pre_hook(module_stats: ModuleStats, module: nn.Module, input):
    if False:
        while True:
            i = 10
    assert not module_stats.done
    module_stats.start_time = time.time()

def _forward_post_hook(module_stats: ModuleStats, module: nn.Module, input, output):
    if False:
        i = 10
        return i + 15
    assert not module_stats.done
    module_stats.end_time = time.time()
    module_stats.duration = module_stats.end_time - module_stats.start_time
    (inputs, outputs) = (_flatten(input), _flatten(output))
    module_stats.input_shape = inputs[0].size()
    module_stats.output_shape = outputs[0].size()
    parameter_quantity = 0
    for (name, p) in module.named_parameters():
        parameter_quantity += 0 if p is None else torch.numel(p.data)
    module_stats.parameter_quantity = parameter_quantity
    inference_memory = 1
    for oi in outputs:
        for s in oi.size():
            inference_memory *= s
    inference_memory = inference_memory * 4 / 1024 ** 2
    module_stats.inference_memory = inference_memory
    module_stats.MAdd = compute_madd(module, inputs, outputs)
    module_stats.Flops = compute_flops(module, inputs, outputs)
    module_stats.Memory = compute_memory(module, inputs, outputs)
    return output

def get_parent_node(root_node, stat_node_name):
    if False:
        return 10
    assert isinstance(root_node, StatNode)
    node = root_node
    names = stat_node_name.split('.')
    for i in range(len(names) - 1):
        node_name = '.'.join(names[0:i + 1])
        child_index = node.find_child_index(node_name)
        assert child_index != -1
        node = node.children[child_index]
    return node

def _convert_leaf_modules_to_stat_tree(leaf_modules):
    if False:
        return 10
    assert isinstance(leaf_modules, OrderedDict)
    create_index = 1
    root_node = StatNode(name='root', parent=None)
    for (name, module_stats) in leaf_modules.items():
        names = name.split('.')
        for i in range(len(names)):
            create_index += 1
            stat_node_name = '.'.join(names[0:i + 1])
            parent_node = get_parent_node(root_node, stat_node_name)
            node = StatNode(name=stat_node_name, parent=parent_node)
            parent_node.add_child(node)
            if i == len(names) - 1:
                input_shape = module_stats.input_shape
                output_shape = module_stats.output_shape
                node.input_shape = input_shape
                node.output_shape = output_shape
                node.parameter_quantity = module_stats.parameter_quantity
                node.inference_memory = module_stats.inference_memory
                node.MAdd = module_stats.MAdd
                node.Flops = module_stats.Flops
                node.duration = module_stats.duration
                node.Memory = module_stats.Memory
    return StatTree(root_node)