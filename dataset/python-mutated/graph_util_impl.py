"""Helpers to manipulate a tensor graph in python.
"""
import copy
import re
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import _proto_comparators
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
GraphDef = tf_export(v1=['GraphDef'])(graph_pb2.GraphDef)
_VARIABLE_OPS = {'Assign', 'AssignAdd', 'AssignSub', 'Queue', 'ScatterAdd', 'ScatterSub', 'ScatterUpdate', 'TruncatedNormal', 'Variable', 'VariableV2'}
_CONTROL_FLOW_OP_NAMES_OR_IDENTITY = ['Switch', 'Enter', 'Exit', 'Identity', 'Merge', 'NextIteration']
_DEPRECATION_MSG = 'This API was designed for TensorFlow v1. See https://www.tensorflow.org/guide/migrate for instructions on how to migrate your code to TensorFlow v2.'

def _is_variable_op(op):
    if False:
        return 10
    "Returns true if 'op' refers to a Variable node."
    return op in _VARIABLE_OPS
graph_pb2.GraphDef.__doc__ = "A protobuf containing the graph of operations.\n\n@compatibility(TF2)\nThis API is not available in TensorFlow 2.x.\n\nYou should not need to use `GraphDef`s directly in TF2. To load `GraphDef`s in\nTF2, use SavedModel. The SavedModel contains the `GraphDef`.\n\nBefore:\n\n```python\nwith tf.io.gfile.GFile('/tmp/graph.pb', 'rb') as f:\n  graph_def = tf.compat.v1.GraphDef()\n  graph_def.ParseFromString(f.read())\n```\n\nAfter:\n\n```python\ntf.saved_model.load('/tmp/saved_model')\n```\n\nIf you would like to create a `GraphDef` in TF2, use `tf.function` and\n`get_concrete_function`.\n\n>>> @tf.function\n>>> def f(x):\n>>>   return x\n>>>\n>>> graph_def = f.get_concrete_function(1.).graph.as_graph_def()\n>>> print(graph_def)\n\n@end_compatibility\n\n"

@deprecation.deprecated(date=None, instructions=_DEPRECATION_MSG)
@tf_export(v1=['graph_util.must_run_on_cpu'])
def must_run_on_cpu(node, pin_variables_on_cpu=False):
    if False:
        i = 10
        return i + 15
    'Returns True if the given node_def must run on CPU, otherwise False.\n\n  Args:\n    node: The node to be assigned to a device. Could be either an ops.Operation\n      or NodeDef.\n    pin_variables_on_cpu: If True, this function will return False if node_def\n      represents a variable-related op.\n\n  Returns:\n    True if the given node must run on CPU, otherwise False.\n  '
    if isinstance(node, ops.Operation):
        node_def = node.node_def
    else:
        assert isinstance(node, node_def_pb2.NodeDef)
        node_def = node
    if pin_variables_on_cpu and _is_variable_op(node_def.op):
        return True
    if node_def.op == 'Const':
        dtype = node_def.attr['dtype'].type
        if dtype == dtypes.string or dtype == dtypes.int32:
            return True
    if node_def.op in ['DynamicStitch', 'ParallelDynamicStitch']:
        dtype = node_def.attr['T'].type
        if dtype == dtypes.int32:
            return True
    if node_def.op in ['Cast']:
        dtype = node_def.attr['SrcT'].type
        if dtype == dtypes.int32:
            return True
    return False

def _node_name(n):
    if False:
        for i in range(10):
            print('nop')
    if n.startswith('^'):
        return n[1:]
    else:
        return n.split(':')[0]

def _get_colocated_node_name(colocated_node_name):
    if False:
        for i in range(10):
            print('nop')
    'Decodes colocated node name and returns it without loc:@ prepended.'
    colocated_node_decoded = colocated_node_name.decode('utf-8')
    if colocated_node_decoded.startswith('loc:@'):
        return colocated_node_decoded[5:]
    return colocated_node_decoded

def _extract_graph_summary(graph_def):
    if False:
        i = 10
        return i + 15
    'Extracts useful information from the graph and returns them.'
    name_to_input_name = {}
    name_to_node = {}
    name_to_seq_num = {}
    seq = 0
    for node in graph_def.node:
        n = _node_name(node.name)
        name_to_node[n] = node
        name_to_input_name[n] = [_node_name(x) for x in node.input]
        if '_class' in node.attr:
            for colocated_node_name in node.attr['_class'].list.s:
                name_to_input_name[n].append(_get_colocated_node_name(colocated_node_name))
        name_to_seq_num[n] = seq
        seq += 1
    return (name_to_input_name, name_to_node, name_to_seq_num)

def _assert_nodes_are_present(name_to_node, nodes):
    if False:
        while True:
            i = 10
    'Assert that nodes are present in the graph.'
    for d in nodes:
        assert d in name_to_node, '%s is not in graph' % d

def _bfs_for_reachable_nodes(target_nodes, name_to_input_name):
    if False:
        return 10
    'Breadth first search for reachable nodes from target nodes.'
    nodes_to_keep = set()
    next_to_visit = list(target_nodes)
    while next_to_visit:
        node = next_to_visit[0]
        del next_to_visit[0]
        if node in nodes_to_keep:
            continue
        nodes_to_keep.add(node)
        if node in name_to_input_name:
            next_to_visit += name_to_input_name[node]
    return nodes_to_keep

@deprecation.deprecated(date=None, instructions=_DEPRECATION_MSG)
@tf_export(v1=['graph_util.extract_sub_graph'])
def extract_sub_graph(graph_def, dest_nodes):
    if False:
        for i in range(10):
            print('nop')
    "Extract the subgraph that can reach any of the nodes in 'dest_nodes'.\n\n  Args:\n    graph_def: A graph_pb2.GraphDef proto.\n    dest_nodes: An iterable of strings specifying the destination node names.\n  Returns:\n    The GraphDef of the sub-graph.\n\n  Raises:\n    TypeError: If 'graph_def' is not a graph_pb2.GraphDef proto.\n  "
    if not isinstance(graph_def, graph_pb2.GraphDef):
        raise TypeError(f'graph_def must be a graph_pb2.GraphDef proto, but got type {type(graph_def)}.')
    if isinstance(dest_nodes, str):
        raise TypeError(f'dest_nodes must be an iterable of strings, but got type {type(dest_nodes)}.')
    (name_to_input_name, name_to_node, name_to_seq_num) = _extract_graph_summary(graph_def)
    _assert_nodes_are_present(name_to_node, dest_nodes)
    nodes_to_keep = _bfs_for_reachable_nodes(dest_nodes, name_to_input_name)
    nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: name_to_seq_num[n])
    out = graph_pb2.GraphDef()
    for n in nodes_to_keep_list:
        out.node.extend([copy.deepcopy(name_to_node[n])])
    out.library.CopyFrom(graph_def.library)
    out.versions.CopyFrom(graph_def.versions)
    return out

@deprecation.deprecated(date=None, instructions=_DEPRECATION_MSG)
@tf_export(v1=['graph_util.tensor_shape_from_node_def_name'])
def tensor_shape_from_node_def_name(graph, input_name):
    if False:
        print('Hello World!')
    "Convenience function to get a shape from a NodeDef's input string."
    if ':' not in input_name:
        canonical_name = input_name + ':0'
    else:
        canonical_name = input_name
    tensor = graph.get_tensor_by_name(canonical_name)
    shape = tensor.get_shape()
    return shape

@deprecation.deprecated(date=None, instructions=_DEPRECATION_MSG)
@tf_export(v1=['graph_util.remove_training_nodes'])
def remove_training_nodes(input_graph, protected_nodes=None):
    if False:
        print('Hello World!')
    "Prunes out nodes that aren't needed for inference.\n\n  There are nodes like Identity and CheckNumerics that are only useful\n  during training, and can be removed in graphs that will be used for\n  nothing but inference. Here we identify and remove them, returning an\n  equivalent graph. To be specific, CheckNumerics nodes are always removed, and\n  Identity nodes that aren't involved in control edges are spliced out so that\n  their input and outputs are directly connected.\n\n  Args:\n    input_graph: Model to analyze and prune.\n    protected_nodes: An optional list of names of nodes to be kept\n      unconditionally. This is for example useful to preserve Identity output\n      nodes.\n\n  Returns:\n    A list of nodes with the unnecessary ones removed.\n  "
    if not protected_nodes:
        protected_nodes = []
    types_to_remove = {'CheckNumerics': True}
    input_nodes = input_graph.node
    names_to_remove = {}
    for node in input_nodes:
        if node.op in types_to_remove and node.name not in protected_nodes:
            names_to_remove[node.name] = True
    nodes_after_removal = []
    for node in input_nodes:
        if node.name in names_to_remove:
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        del new_node.input[:]
        for full_input_name in input_before_removal:
            input_name = re.sub('^\\^', '', full_input_name)
            if input_name in names_to_remove:
                continue
            new_node.input.append(full_input_name)
        nodes_after_removal.append(new_node)
    types_to_splice = {'Identity': True}
    control_input_names = set()
    node_names_with_control_input = set()
    node_in_colocated = set()
    for node in nodes_after_removal:
        for node_input in node.input:
            if '^' in node_input:
                control_input_names.add(node_input.replace('^', ''))
                node_names_with_control_input.add(node.name)
        if '_class' in node.attr:
            for colocated_node_name in node.attr['_class'].list.s:
                node_in_colocated.add(_get_colocated_node_name(colocated_node_name))
    names_to_splice = {}
    for node in nodes_after_removal:
        if node.op in types_to_splice and node.name not in protected_nodes:
            if node.name in node_in_colocated:
                continue
            if node.name not in node_names_with_control_input:
                names_to_splice[node.name] = node.input[0]
    names_to_splice = {name: value for (name, value) in names_to_splice.items() if name not in control_input_names}
    nodes_after_splicing = []
    for node in nodes_after_removal:
        if node.name in names_to_splice:
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        del new_node.input[:]
        for full_input_name in input_before_removal:
            input_name = re.sub('^\\^', '', full_input_name)
            while input_name in names_to_splice:
                full_input_name = names_to_splice[input_name]
                input_name = re.sub('^\\^', '', full_input_name)
            new_node.input.append(full_input_name)
        nodes_after_splicing.append(new_node)
    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(nodes_after_splicing)
    return output_graph

@tf_export('__internal__.graph_util.graph_defs_equal', v1=[])
def graph_defs_equal(graph_def_1: graph_pb2.GraphDef, graph_def_2: graph_pb2.GraphDef, treat_nan_as_equal: bool=False) -> bool:
    if False:
        while True:
            i = 10
    "Returns True iff the graph def arguments are structurally equivalent.\n\n  The notion of equivalence encoded here checks that the set of NodeDefs in\n  the GraphDef's function library and main graph body are identical.\n  Additionally, it checks that the functions in the function library are equal\n  as sets.\n\n  Example usage:\n\n  ```\n  with tf.Graph().as_default() as g1:\n    tf.constant(1)\n\n  with tf.Graph().as_default() as g2:\n    tf.constant(2)\n\n  with tf.Graph().as_default() as g3:\n    tf.constant(1)\n\n  assert tf.__internal__.graph_util.graph_defs_equal(g1.as_graph_def(),\n                                                     g3.as_graph_def())\n\n  assert not tf.__internal__.graph_util.graph_defs_equal(g1.as_graph_def(),\n                                                         g2.as_graph_def())\n  ```\n\n  Args:\n    graph_def_1: Instance of `graph_pb2.GraphDef` to compare.\n    graph_def_2: Instance of `graph_pb2.GraphDef` to compare.\n    treat_nan_as_equal: Boolean indicating whether or not to treat nan\n      floating-point values as equal. This is crucial for any equivalence\n      relation defined over GraphDefs, to ensure symmetry.\n\n  Returns:\n    Boolean indicating structural equivalence as described above.\n\n  Raises:\n    TypeError: If either of the GraphDefs are not instances of\n      `graph_pb2.GraphDef`.\n  "
    if not isinstance(graph_def_1, graph_pb2.GraphDef):
        raise TypeError(f'graph_def_1 must be a graph_pb2.GraphDef proto, but got type {type(graph_def_1)}.')
    if not isinstance(graph_def_2, graph_pb2.GraphDef):
        raise TypeError(f'graph_def_2 must be a graph_pb2.GraphDef proto, but got type {type(graph_def_2)}.')
    options = _proto_comparators.ProtoComparisonOptions(treat_nan_as_equal)
    return _proto_comparators.EqualsGraphDef(graph_def_1.SerializeToString(), graph_def_2.SerializeToString(), options)