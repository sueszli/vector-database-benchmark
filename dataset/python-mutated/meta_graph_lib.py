"""MetaGraph lib provides utilities to manipulate MetaGraphDefs.

This is an internal Hub utility and not part of the public API.
"""
import re
from absl import logging
import tensorflow as tf

def prepend_name_scope(name, import_scope):
    if False:
        while True:
            i = 10
    'Prepends name scope to a name.'
    if import_scope:
        try:
            str_to_replace = '([\\^]|loc:@|^)(.*)'
            return re.sub(str_to_replace, '\\1' + import_scope + '/\\2', tf.compat.as_str_any(name))
        except TypeError as e:
            logging.warning(e)
            return name
    else:
        return name

def prefix_shared_name_attributes(meta_graph, absolute_import_scope):
    if False:
        i = 10
        return i + 15
    'In-place prefixes shared_name attributes of nodes.'
    shared_name_attr = 'shared_name'
    for node in meta_graph.graph_def.node:
        shared_name_value = node.attr.get(shared_name_attr, None)
        if shared_name_value and shared_name_value.HasField('s'):
            if shared_name_value.s:
                node.attr[shared_name_attr].s = tf.compat.as_bytes(prepend_name_scope(shared_name_value.s, import_scope=absolute_import_scope))

def mark_backward(output_tensor):
    if False:
        i = 10
        return i + 15
    "Function to propagate backwards in the graph and mark nodes as used.\n\n  Traverses iteratively through the graph from the end tensor, through the op\n  that generates the tensor, and then to the input tensors that feed the op.\n  Nodes encountered are stored in used_node_names.\n\n  Args:\n    output_tensor: A Tensor which we start the propagation.\n  Returns:\n    used_node_names: A set of strings, stores the name of nodes we've marked as\n      visited.\n  "
    used_node_names = set()
    tensors = [output_tensor]
    while tensors:
        op = tensors.pop().op
        if op.name in used_node_names:
            continue
        used_node_names.add(op.name)
        tensors.extend(op.inputs)
        for control_input_op in op.control_inputs:
            used_node_names.add(control_input_op.name)
            tensors.extend(control_input_op.inputs)
    return used_node_names

def prune_unused_nodes(meta_graph, signature_def):
    if False:
        for i in range(10):
            print('nop')
    'Function to prune unused ops given a signature def.\n\n  This function does a graph traversal through from all outputs as\n  defined in the signature_def to collect all used nodes. Then, any\n  nodes which are unused can be discarded. This is useful for graph which are\n  executing eagerly or on TPUs.\n\n  Args:\n    meta_graph: The input/output MetaGraphDef for which we wish to prune.\n   signature_def: A SignatureDef which specifies the outputs from which we wish\n     to start graph traversal.\n  '
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        tf.compat.v1.train.import_meta_graph(meta_graph, input_map={}, import_scope='')
        used_node_names = set()
        for (_, tensor_def) in signature_def.outputs.items():
            output_tensor = graph.get_tensor_by_name(tensor_def.name)
            used_node_names |= mark_backward(output_tensor)
        node_filter_in_list = []
        for node in meta_graph.graph_def.node:
            if node.name in used_node_names or node.op == 'VarHandleOp':
                node_filter_in_list.append(node)
        del meta_graph.graph_def.node[:]
        meta_graph.graph_def.node.extend(node_filter_in_list)
    del graph

def prune_feed_map(meta_graph, feed_map):
    if False:
        for i in range(10):
            print('nop')
    'Function to prune the feedmap of nodes which no longer exist.'
    node_names = [x.name + ':0' for x in meta_graph.graph_def.node]
    keys_to_delete = []
    for (k, _) in feed_map.items():
        if k not in node_names:
            keys_to_delete.append(k)
    for k in keys_to_delete:
        del feed_map[k]

def filter_collections(meta_graph, collections):
    if False:
        print('Hello World!')
    collections = frozenset(collections)
    for name in list(meta_graph.collection_def.keys()):
        if name not in collections:
            del meta_graph.collection_def[name]