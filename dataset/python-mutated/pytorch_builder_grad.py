"""
    File name: plot-pytorch-autograd-graph.py
    Author: Ludovic Trottier
    Date created: November 8, 2017.
    Date last modified: November 8, 2017
    Credits: moskomule (https://discuss.pytorch.org/t/print-autograd-graph/692/15)
"""
from graphviz import Digraph
import torch
from torch.autograd import Variable
from . import transforms as ht
from collections import abc
import numpy as np
from .graph import Graph, Node
FRAMEWORK_TRANSFORMS = [ht.Rename(op='onnx::(.*)', to='\\1'), ht.Rename(op='Gemm', to='Linear'), ht.Rename(op='aten::max\\_pool2d\\_with\\_indices', to='MaxPool'), ht.Rename(op='BatchNormalization', to='BatchNorm')]

def add_node2dot(dot, var, id, label, op=None, output_shape=None, params=None):
    if False:
        return 10
    hl_node = Node(uid=id, name=op, op=label, output_shape=output_shape, params=params)
    dot.add_node(hl_node)

def make_dot(var, params, dot):
    if False:
        print('Hello World!')
    ' Produces Graphviz representation of PyTorch autograd graph.\n    \n    Blue nodes are trainable Variables (weights, bias).\n    Orange node are saved tensors for the backward pass.\n    \n    Args:\n        var: output Variable\n        params: list of (name, Parameters)\n    '
    param_map2 = {k: v for (k, v) in params}
    print(param_map2)
    param_map = {id(v): k for (k, v) in params}
    node_attr = dict(style='filled', shape='box', align='left', fontsize='12', ranksep='0.1', height='0.2')
    seen = set()

    def add_nodes(dot, var):
        if False:
            return 10
        if var not in seen:
            node_id = str(id(var))
            if torch.is_tensor(var):
                node_label = 'saved tensor\n{}'.format(tuple(var.size()))
                add_node2dot(dot, var, node_id, node_label, op=None)
            elif hasattr(var, 'variable'):
                variable_name = param_map.get(id(var.variable))
                variable_size = tuple(var.variable.size())
                node_name = '{}\n{}'.format(variable_name, variable_size)
                add_node2dot(dot, var, node_id, node_name, op=None)
            else:
                node_label = type(var).__name__.replace('Backward', '')
                add_node2dot(dot, var, node_id, node_label, op=None)
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.add_edge_by_id(str(id(u[0])), str(id(var)), None)
                        add_nodes(dot, u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.add_edge_by_id(str(id(t)), str(id(var)), None)
                    add_nodes(dot, t)
    add_nodes(dot, var.grad_fn)
    return dot

def import_graph(hl_graph, model, args, input_names=None, verbose=False):
    if False:
        i = 10
        return i + 15
    if args is None:
        args = [1, 3, 224, 224]
    if not isinstance(args, torch.Tensor) and hasattr(args, '__len__') and hasattr(args, '__getitem__') and (not isinstance(args, (str, abc.ByteString))):
        args = torch.ones(args)
    y = model(args)
    g = make_dot(y, model.named_parameters(), hl_graph)
    return hl_graph