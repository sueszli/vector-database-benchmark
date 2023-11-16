"""
HiddenLayer

TensorFlow graph importer.
 
Written by Phil Ferriere. Edits by Waleed Abdulla.
Licensed under the MIT License
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import tensorflow as tf
from .graph import Graph, Node
from . import transforms as ht
FRAMEWORK_TRANSFORMS = [ht.Rename(op='(\\w+)V\\d', to='\\1'), ht.Prune('Const'), ht.Prune('PlaceholderWithDefault'), ht.Prune('Variable'), ht.Prune('VarIsInitializedOp'), ht.Prune('VarHandleOp'), ht.Prune('ReadVariableOp'), ht.PruneBranch('Assign'), ht.PruneBranch('AssignSub'), ht.PruneBranch('AssignAdd'), ht.PruneBranch('AssignVariableOp'), ht.Prune('ApplyMomentum'), ht.Prune('ApplyAdam'), ht.FoldId('^(gradients)/.*', 'NoOp'), ht.Prune('NoOp'), ht.Rename(op='DepthwiseConv2dNative', to='SeparableConv'), ht.Rename(op='Conv2D', to='Conv'), ht.Rename(op='FusedBatchNorm', to='BatchNorm'), ht.Rename(op='MatMul', to='Linear'), ht.Fold('Conv > BiasAdd', '__first__'), ht.Fold('Linear > BiasAdd', '__first__'), ht.Fold('Shape > StridedSlice > Pack > Reshape', '__last__'), ht.FoldId('(.+)/dropout/.*', 'Dropout'), ht.FoldId('(softmax_cross\\_entropy)\\_with\\_logits.*', 'SoftmaxCrossEntropy')]

def dump_tf_graph(tfgraph, tfgraphdef):
    if False:
        i = 10
        return i + 15
    'List all the nodes in a TF graph.\n    tfgraph: A TF Graph object.\n    tfgraphdef: A TF GraphDef object.\n    '
    print('Nodes ({})'.format(len(tfgraphdef.node)))
    f = '{:15} {:59} {:20} {}'
    print(f.format('kind', 'scopeName', 'shape', 'inputs'))
    for node in tfgraphdef.node:
        scopename = node.name
        kind = node.op
        inputs = node.input
        shape = tf.graph_util.tensor_shape_from_node_def_name(tfgraph, scopename)
        print(f.format(kind, scopename, str(shape), inputs))

def import_graph(hl_graph, tf_graph, output=None, verbose=False):
    if False:
        return 10
    'Convert TF graph to directed graph\n    tfgraph: A TF Graph object.\n    output: Name of the output node (string).\n    verbose: Set to True for debug print output\n    '
    graph_def = tf_graph.as_graph_def(add_shapes=True)
    graph_def = tf.graph_util.remove_training_nodes(graph_def)
    if verbose:
        dump_tf_graph(tf_graph, graph_def)
    for tf_node in graph_def.node:
        try:
            (op, uid, name, shape, params) = import_node(tf_node, tf_graph, verbose)
        except:
            if verbose:
                logging.exception('Failed to read node {}'.format(tf_node))
            continue
        hl_node = Node(uid=uid, name=name, op=op, output_shape=shape, params=params)
        hl_graph.add_node(hl_node)
        for target_node in graph_def.node:
            target_inputs = target_node.input
            if uid in target_node.input:
                hl_graph.add_edge_by_id(uid, target_node.name, shape)
    return hl_graph

def import_node(tf_node, tf_graph, verbose=False):
    if False:
        for i in range(10):
            print('nop')
    op = tf_node.op
    uid = tf_node.name
    name = None
    shape = None
    if tf_node.op != 'NoOp':
        try:
            shape = tf.graph_util.tensor_shape_from_node_def_name(tf_graph, tf_node.name)
            if shape.ndims is not None:
                shape = shape.as_list()
        except:
            if verbose:
                logging.exception('Error reading shape of {}'.format(tf_node.name))
    params = {}
    if op == 'Conv2D' or op == 'DepthwiseConv2dNative':
        kernel_shape = tf.graph_util.tensor_shape_from_node_def_name(tf_graph, tf_node.input[1])
        kernel_shape = [int(a) for a in kernel_shape]
        params['kernel_shape'] = kernel_shape[0:2]
        if 'strides' in tf_node.attr.keys():
            strides = [int(a) for a in tf_node.attr['strides'].list.i]
            params['stride'] = strides[1:3]
    elif op == 'MaxPool' or op == 'AvgPool':
        if 'ksize' in tf_node.attr.keys():
            kernel_shape = [int(a) for a in tf_node.attr['ksize'].list.i]
            params['kernel_shape'] = kernel_shape[1:3]
        if 'strides' in tf_node.attr.keys():
            strides = [int(a) for a in tf_node.attr['strides'].list.i]
            params['stride'] = strides[1:3]
    return (op, uid, name, shape, params)