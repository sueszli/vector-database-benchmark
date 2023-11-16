from torch.utils.tensorboard._pytorch_graph import GraphPy, NodePyIO, NodePyOP
import torch
from . import transforms as ht
from collections import abc
import numpy as np
FRAMEWORK_TRANSFORMS = [ht.Rename(op='onnx::(.*)', to='\\1'), ht.Rename(op='Gemm', to='Linear'), ht.Rename(op='aten::max\\_pool2d\\_with\\_indices', to='MaxPool'), ht.Rename(op='BatchNormalization', to='BatchNorm')]

def parse(graph, args=None, omit_useless_nodes=True):
    if False:
        return 10
    'This method parses an optimized PyTorch model graph and produces\n    a list of nodes and node stats for eventual conversion to TensorBoard\n    protobuf format.\n    Args:\n      graph (PyTorch module): The model to be parsed.\n      args (tuple): input tensor[s] for the model.\n      omit_useless_nodes (boolean): Whether to remove nodes from the graph.\n    '
    n_inputs = len(args)
    scope = {}
    nodes_py = GraphPy()
    for (i, node) in enumerate(graph.inputs()):
        if omit_useless_nodes:
            if len(node.uses()) == 0:
                continue
        if i < n_inputs:
            nodes_py.append(NodePyIO(node, 'input'))
        else:
            nodes_py.append(NodePyIO(node))
    for node in graph.nodes():
        nodes_py.append(NodePyOP(node))
    for node in graph.outputs():
        NodePyIO(node, 'output')
    nodes_py.find_common_root()
    nodes_py.populate_namespace_from_OP_to_IO()
    return nodes_py

def graph(model, args, verbose=False):
    if False:
        i = 10
        return i + 15
    '\n    This method processes a PyTorch model and produces a `GraphDef` proto\n    that can be logged to TensorBoard.\n    Args:\n      model (PyTorch module): The model to be parsed.\n      args (tuple): input tensor[s] for the model.\n      verbose (bool): Whether to print out verbose information while\n        processing.\n    '
    with torch.onnx.set_training(model, False):
        try:
            trace = torch.jit.trace(model, args)
            graph = trace.graph
        except RuntimeError as e:
            print(e)
            print('Error occurs, No graph saved')
            raise e
    if verbose:
        print(graph)
    return parse(graph, args)

def import_graph(hl_graph, model, args, input_names=None, verbose=False):
    if False:
        print('Hello World!')
    if args is None:
        args = [1, 3, 224, 224]
    if not isinstance(args, torch.Tensor) and hasattr(args, '__len__') and hasattr(args, '__getitem__') and (not isinstance(args, (str, abc.ByteString))):
        args = torch.ones(args)
    graph_py = graph(model, args, verbose)
    return hl_graph