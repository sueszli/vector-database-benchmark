"""Removes parts of a graph that are only needed for training.

There are several common transformations that can be applied to GraphDefs
created to train a model, that help reduce the amount of computation needed when
the network is used only for inference. These include:

 - Removing training-only operations like checkpoint saving.

 - Stripping out parts of the graph that are never reached.

 - Removing debug operations like CheckNumerics.

 - Folding batch normalization ops into the pre-calculated weights.

 - Fusing common operations into unified versions.

This script takes a frozen GraphDef file (where the weight variables have been
converted into constants by the freeze_graph script) and outputs a new GraphDef
with the optimizations applied.

An example of command-line usage is:

bazel build tensorflow/python/tools:optimize_for_inference && \\
bazel-bin/tensorflow/python/tools/optimize_for_inference \\
--input_graph=some_graph_def.pb \\
--output_graph=/tmp/optimized_graph.pb \\
--input_names=Mul \\
--output_names=softmax

"""
import collections
import math
import re
from typing import Mapping, Sequence
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import flags as flags_lib
from tensorflow.python.platform import tf_logging
from tensorflow.python.tools import strip_unused_lib
flags = flags_lib
FLAGS = flags.FLAGS
INPUT_ORDER = {'BatchNormWithGlobalNormalization': ['conv_op', 'mean_op', 'var_op', 'beta_op', 'gamma_op'], 'FusedBatchNorm': ['conv_op', 'gamma_op', 'beta_op', 'mean_op', 'var_op'], 'FusedBatchNormV3': ['conv_op', 'gamma_op', 'beta_op', 'mean_op', 'var_op']}
EPSILON_ATTR = {'BatchNormWithGlobalNormalization': 'variance_epsilon', 'FusedBatchNorm': 'epsilon', 'FusedBatchNormV3': 'epsilon'}
PLACEHOLDER_WITH_DEFAULT_LIST = {'keras_learning_phase': 'False'}

def optimize_for_inference(input_graph_def: graph_pb2.GraphDef, input_node_names: Sequence[str], output_node_names: Sequence[str], placeholder_type_enum: int, toco_compatible: bool=False, placeholder_to_const_names=None) -> graph_pb2.GraphDef:
    if False:
        while True:
            i = 10
    'Applies a series of inference optimizations on the input graph.\n\n  Args:\n    input_graph_def: A GraphDef containing a training model.\n    input_node_names: A list of names of the nodes that are fed inputs during\n      inference.\n    output_node_names: A list of names of the nodes that produce the final\n      results.\n    placeholder_type_enum: The AttrValue enum for the placeholder data type, or\n      a list that specifies one value per input node name.\n    toco_compatible: Boolean, if True, only runs optimizations that result in\n      TOCO compatible graph operations (default=False).\n    placeholder_to_const_names: A list of names of the PlaceholderWithDefault\n      nodes to be converted to Constant.\n\n  Returns:\n    An optimized version of the input graph.\n  '
    ensure_graph_is_valid(input_graph_def)
    optimized_graph_def = input_graph_def
    optimized_graph_def = convert_placeholder_to_const(optimized_graph_def, placeholder_to_const_names)
    optimized_graph_def = strip_unused_lib.strip_unused(optimized_graph_def, input_node_names, output_node_names, placeholder_type_enum)
    optimized_graph_def = graph_util.remove_training_nodes(optimized_graph_def, output_node_names)
    optimized_graph_def = fold_batch_norms(optimized_graph_def)
    if not toco_compatible:
        optimized_graph_def = fuse_resize_and_conv(optimized_graph_def, output_node_names)
    ensure_graph_is_valid(optimized_graph_def)
    return optimized_graph_def

def strtobool(val_str):
    if False:
        print('Hello World!')
    "Return boolean value of it's equivalent string representation"
    if val_str in ('True', 'true'):
        return True
    elif val_str in ('False', 'false'):
        return False
    else:
        tf_logging.warning('Wrong string values.       Supports False/false or True/true only. val_str = ', val_str)
        return False

def parse_entry(entry):
    if False:
        return 10
    'Parse a "key=value" pair separated by \'=\'\n\n  eg: var_name=False\n  '
    items = entry.split('=')
    key = items[0].strip()
    if len(items) > 1:
        value = items[1]
        return (key, value)
    else:
        return (None, None)

def parse_nodes_dict(nodes):
    if False:
        return 10
    'Parse a series of key-value pairs and return a dictionary'
    d = {}
    if nodes:
        for node in nodes:
            (key, val) = parse_entry(node)
            if key is not None:
                d[key] = val
    return d

def ensure_graph_is_valid(graph_def: graph_pb2.GraphDef) -> None:
    if False:
        print('Hello World!')
    'Makes sure that the graph is internally consistent.\n\n  Checks basic properties of the graph def and raises an exception if there are\n  input references to missing nodes, duplicated names, or other logic errors.\n\n  Args:\n    graph_def: Definition of a graph to be checked.\n\n  Raises:\n    ValueError: If the graph is incorrectly constructed.\n  '
    node_map = {}
    for node in graph_def.node:
        if node.name not in node_map:
            node_map[node.name] = node
        else:
            raise ValueError('Duplicate node names detected for ', node.name)
    for node in graph_def.node:
        for input_name in node.input:
            input_node_name = node_name_from_input(input_name)
            if input_node_name not in node_map:
                raise ValueError('Input for ', node.name, ' not found: ', input_name)

def node_name_from_input(node_name: str) -> str:
    if False:
        print('Hello World!')
    'Strips off ports and other decorations to get the underlying node name.'
    if node_name.startswith('^'):
        node_name = node_name[1:]
    m = re.search('(.*):\\d+$', node_name)
    if m:
        node_name = m.group(1)
    return node_name

def node_from_map(node_map: Mapping[str, node_def_pb2.NodeDef], name: str) -> node_def_pb2.NodeDef:
    if False:
        for i in range(10):
            print('nop')
    "Pulls a node def from a dictionary for a given name.\n\n  Args:\n    node_map: Dictionary containing an entry indexed by name for every node.\n    name: Identifies the node we want to find.\n\n  Returns:\n    NodeDef of the node with the given name.\n\n  Raises:\n    ValueError: If the node isn't present in the dictionary.\n  "
    stripped_name = node_name_from_input(name)
    if stripped_name not in node_map:
        raise ValueError("No node named '%s' found in map." % name)
    return node_map[stripped_name]

def values_from_const(node_def: node_def_pb2.NodeDef) -> np.ndarray:
    if False:
        while True:
            i = 10
    "Extracts the values from a const NodeDef as a numpy ndarray.\n\n  Args:\n    node_def: Const NodeDef that has the values we want to access.\n\n  Returns:\n    Numpy ndarray containing the values.\n\n  Raises:\n    ValueError: If the node isn't a Const.\n  "
    if node_def.op != 'Const':
        raise ValueError(f'Can not extract constant value from a node that is not Const. Got:\n{node_def}')
    input_tensor = node_def.attr['value'].tensor
    tensor_value = tensor_util.MakeNdarray(input_tensor)
    return tensor_value

def scale_after_normalization(node: node_def_pb2.NodeDef) -> bool:
    if False:
        while True:
            i = 10
    if node.op == 'BatchNormWithGlobalNormalization':
        return node.attr['scale_after_normalization'].b
    return True

def fold_batch_norms(input_graph_def: graph_pb2.GraphDef) -> graph_pb2.GraphDef:
    if False:
        i = 10
        return i + 15
    "Removes batch normalization ops by folding them into convolutions.\n\n  Batch normalization during training has multiple dynamic parameters that are\n  updated, but once the graph is finalized these become constants. That means\n  there's an opportunity to reduce the computations down to a scale and\n  addition, rather than the more expensive multiple ops, and even bake the\n  scaling into the convolution weights. This function identifies the typical\n  pattern of batch normalization subgraphs, and performs the transformation to\n  fold the computations down into a simpler form. It currently only supports\n  batch normalization that's performed by the BatchNormWithGlobalNormalization\n  FusedBatchNorm and FusedBatchNormV3 ops, and will need to be extended in the\n  future to handle the newer style.\n\n  Args:\n    input_graph_def: A GraphDef containing a model.\n\n  Returns:\n    Modified graph with BN ops removed, and modified weights.\n\n  Raises:\n    ValueError: If the graph is badly formed with duplicate node names.\n  "
    input_node_map = {}
    for node in input_graph_def.node:
        if node.name not in input_node_map:
            input_node_map[node.name] = node
        else:
            raise ValueError('Duplicate node names detected for ', node.name)
    nodes_to_skip = {}
    new_ops = []
    for node in input_graph_def.node:
        if node.op not in ('BatchNormWithGlobalNormalization', 'FusedBatchNorm', 'FusedBatchNormV3'):
            continue
        bias = None
        conv_op = node_from_map(input_node_map, node.input[INPUT_ORDER[node.op].index('conv_op')])
        if conv_op.op in ['BiasAdd', 'Add', 'AddV2']:
            add_op = conv_op
            conv_op = node_from_map(input_node_map, add_op.input[0])
            bias = node_from_map(input_node_map, add_op.input[1])
            if conv_op.op not in ['Conv2D', 'DepthwiseConv2dNative']:
                conv_op = node_from_map(input_node_map, add_op.input[1])
                bias = node_from_map(input_node_map, add_op.input[0])
        if bias and bias.op != 'Const':
            tf_logging.warning("The bias %s after the conv %s was not a constant. Maybe because freeze_graph wasn't run first?" % (bias.name, conv_op.name))
            continue
        if conv_op.op not in ['Conv2D', 'DepthwiseConv2dNative']:
            tf_logging.warning("Didn't find expected Conv2D or DepthwiseConv2dNative input to '%s'" % node.name)
            continue
        weights_op = node_from_map(input_node_map, conv_op.input[1])
        if weights_op.op != 'Const':
            tf_logging.warning("Didn't find expected conv Constant input to '%s', found %s instead. Maybe because freeze_graph wasn't run first?" % (conv_op.name, weights_op))
            continue
        weights = values_from_const(weights_op)
        if conv_op.op == 'Conv2D':
            channel_count = weights.shape[3]
        elif conv_op.op == 'DepthwiseConv2dNative':
            channel_count = weights.shape[2] * weights.shape[3]
        mean_op = node_from_map(input_node_map, node.input[INPUT_ORDER[node.op].index('mean_op')])
        if mean_op.op != 'Const':
            tf_logging.warning("Didn't find expected mean Constant input to '%s', found %s instead. Maybe because freeze_graph wasn't run first?" % (node.name, mean_op))
            continue
        mean_value = values_from_const(mean_op)
        if mean_value.shape != (channel_count,):
            tf_logging.warning('Incorrect shape for mean, found %s, expected %s, for node %s' % (str(mean_value.shape), str((channel_count,)), node.name))
            continue
        if bias is not None:
            mean_value = mean_value - values_from_const(bias)
        var_op = node_from_map(input_node_map, node.input[INPUT_ORDER[node.op].index('var_op')])
        if var_op.op != 'Const':
            tf_logging.warning("Didn't find expected var Constant input to '%s', found %s instead. Maybe because freeze_graph wasn't run first?" % (node.name, var_op))
            continue
        var_value = values_from_const(var_op)
        if var_value.shape != (channel_count,):
            tf_logging.warning('Incorrect shape for var, found %s, expected %s, for node %s' % (str(var_value.shape), str((channel_count,)), node.name))
            continue
        beta_op = node_from_map(input_node_map, node.input[INPUT_ORDER[node.op].index('beta_op')])
        if beta_op.op != 'Const':
            tf_logging.warning("Didn't find expected beta Constant input to '%s', found %s instead. Maybe because freeze_graph wasn't run first?" % (node.name, beta_op))
            continue
        beta_value = values_from_const(beta_op)
        if beta_value.shape != (channel_count,):
            tf_logging.warning('Incorrect shape for beta, found %s, expected %s, for node %s' % (str(beta_value.shape), str((channel_count,)), node.name))
            continue
        gamma_op = node_from_map(input_node_map, node.input[INPUT_ORDER[node.op].index('gamma_op')])
        if gamma_op.op != 'Const':
            tf_logging.warning("Didn't find expected gamma Constant input to '%s', found %s instead. Maybe because freeze_graph wasn't run first?" % (node.name, gamma_op))
            continue
        gamma_value = values_from_const(gamma_op)
        if gamma_value.shape != (channel_count,):
            tf_logging.warning('Incorrect shape for gamma, found %s, expected %s, for node %s' % (str(gamma_value.shape), str((channel_count,)), node.name))
            continue
        variance_epsilon_value = node.attr[EPSILON_ATTR[node.op]].f
        nodes_to_skip[node.name] = True
        nodes_to_skip[weights_op.name] = True
        nodes_to_skip[conv_op.name] = True
        if bias is not None:
            nodes_to_skip[add_op.name] = True
        if scale_after_normalization(node):
            scale_value = 1.0 / np.vectorize(math.sqrt)(var_value + variance_epsilon_value) * gamma_value
        else:
            scale_value = 1.0 / np.vectorize(math.sqrt)(var_value + variance_epsilon_value)
        offset_value = -mean_value * scale_value + beta_value
        scaled_weights = np.copy(weights)
        it = np.nditer(scaled_weights, flags=['multi_index'], op_flags=['readwrite'])
        if conv_op.op == 'Conv2D':
            while not it.finished:
                current_scale = scale_value[it.multi_index[3]]
                it[0] *= current_scale
                it.iternext()
        elif conv_op.op == 'DepthwiseConv2dNative':
            channel_multiplier = weights.shape[3]
            while not it.finished:
                current_scale = scale_value[it.multi_index[2] * channel_multiplier + it.multi_index[3]]
                it[0] *= current_scale
                it.iternext()
        scaled_weights_op = node_def_pb2.NodeDef()
        scaled_weights_op.op = 'Const'
        scaled_weights_op.name = conv_op.name + '_weights'
        scaled_weights_op.attr['dtype'].CopyFrom(weights_op.attr['dtype'])
        scaled_weights_op.attr['value'].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(scaled_weights, weights.dtype.type, weights.shape)))
        for (i, weights_node) in enumerate(conv_op.input):
            if weights_node == weights_op.name:
                conv_op.input[i] = scaled_weights_op.name
        new_conv_op = node_def_pb2.NodeDef()
        new_conv_op.CopyFrom(conv_op)
        offset_op = node_def_pb2.NodeDef()
        offset_op.op = 'Const'
        offset_op.name = conv_op.name + '_bn_offset'
        offset_op.attr['dtype'].CopyFrom(mean_op.attr['dtype'])
        offset_op.attr['value'].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(offset_value, mean_value.dtype.type, offset_value.shape)))
        bias_add_op = node_def_pb2.NodeDef()
        bias_add_op.op = 'BiasAdd'
        bias_add_op.name = node.name
        bias_add_op.attr['T'].CopyFrom(conv_op.attr['T'])
        bias_add_op.attr['data_format'].CopyFrom(conv_op.attr['data_format'])
        bias_add_op.input.extend([new_conv_op.name, offset_op.name])
        new_ops.extend([scaled_weights_op, new_conv_op, offset_op, bias_add_op])
    result_graph_def = graph_pb2.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        retained_input = []
        for input_node in new_node.input:
            if not input_node.startswith('^') or input_node[1:] not in nodes_to_skip:
                retained_input.append(input_node)
        new_node.input[:] = retained_input
        result_graph_def.node.extend([new_node])
    result_graph_def.node.extend(new_ops)
    result_graph_def.versions.CopyFrom(input_graph_def.versions)
    return result_graph_def

def fuse_resize_and_conv(input_graph_def: graph_pb2.GraphDef, output_node_names: Sequence[str]) -> graph_pb2.GraphDef:
    if False:
        return 10
    "Merges preceding resize and mirror pad ops into a specialized convolution.\n\n  There's a common pattern of enlarging the input to a convolution using a\n  resize operation, and also using MirrorPad to extend the boundaries to that\n  zero edge pixels don't bleed inwards when convolving. This routine looks for\n  that pattern of operations, and fuses them together into a Conv2DWithResizeOp.\n\n  Args:\n    input_graph_def: A GraphDef containing a model.\n    output_node_names: A list of names of the nodes that produce the final\n      results.\n\n  Returns:\n    Modified graph with resize and pad ops merged.\n\n  Raises:\n    ValueError: If the graph is badly formed with duplicate node names.\n  "
    input_node_map = {}
    for node in input_graph_def.node:
        if node.name not in input_node_map:
            input_node_map[node.name] = node
        else:
            raise ValueError('Duplicate node names detected for ', node.name)
    node_reference_count = collections.defaultdict(int)
    for node in input_graph_def.node:
        for input_name in node.input:
            stripped_name = node_name_from_input(input_name)
            node_reference_count[stripped_name] += 1
    for output_name in output_node_names:
        node_reference_count[output_name] += 1
    new_ops = []
    for node in input_graph_def.node:
        if node.op != 'Conv2D':
            continue
        conv_op = node
        input_op = node_from_map(input_node_map, conv_op.input[0])
        if input_op.op == 'MirrorPad':
            mirror_pad_op = input_op
            resize_op = node_from_map(input_node_map, mirror_pad_op.input[0])
            if resize_op.op != 'ResizeBilinear':
                resize_op = None
        else:
            mirror_pad_op = None
            if input_op.op == 'ResizeBilinear':
                resize_op = input_op
            else:
                resize_op = None
        if not mirror_pad_op and (not resize_op):
            continue
        node_reference_count[conv_op.name] = 0
        if mirror_pad_op:
            node_reference_count[mirror_pad_op.name] -= 1
        if resize_op:
            node_reference_count[resize_op.name] -= 1
        fused_conv_op = node_def_pb2.NodeDef()
        if resize_op:
            fused_conv_op.op = 'FusedResizeAndPadConv2D'
        else:
            fused_conv_op.op = 'FusedPadConv2D'
        fused_conv_op.name = conv_op.name
        if mirror_pad_op:
            mirror_paddings_name = mirror_pad_op.input[1]
            mirror_paddings_mode = mirror_pad_op.attr['mode']
        else:
            paddings_op = node_def_pb2.NodeDef()
            paddings_op.op = 'Const'
            paddings_op.name = conv_op.name + '_dummy_paddings'
            paddings_op.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.int32.as_datatype_enum))
            paddings_op.attr['value'].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto([0, 0, 0, 0, 0, 0, 0, 0], dtypes.int32, [4, 2])))
            new_ops.extend([paddings_op])
            mirror_paddings_name = paddings_op.name
            mirror_paddings_mode = attr_value_pb2.AttrValue(s=b'REFLECT')
        if resize_op:
            fused_conv_op.input.extend([resize_op.input[0], resize_op.input[1], mirror_paddings_name, conv_op.input[1]])
            fused_conv_op.attr['resize_align_corners'].CopyFrom(resize_op.attr['align_corners'])
        else:
            fused_conv_op.input.extend([mirror_pad_op.input[0], mirror_paddings_name, conv_op.input[1]])
        fused_conv_op.attr['T'].CopyFrom(conv_op.attr['T'])
        fused_conv_op.attr['mode'].CopyFrom(mirror_paddings_mode)
        fused_conv_op.attr['strides'].CopyFrom(conv_op.attr['strides'])
        fused_conv_op.attr['padding'].CopyFrom(conv_op.attr['padding'])
        new_ops.extend([fused_conv_op])
    result_graph_def = graph_pb2.GraphDef()
    for node in input_graph_def.node:
        if node_reference_count[node.name] < 1:
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        result_graph_def.node.extend([new_node])
    result_graph_def.node.extend(new_ops)
    return result_graph_def

def convert_placeholder_to_const(input_graph_def, nodes_to_convert=None):
    if False:
        while True:
            i = 10
    'Rename the PlaceHolderWithDefault node to constant\n\n  In a frozen graph, PlaceholderWithDefault nodes can be converted to\n  Constant op nodes with same value. This will help simplify the graph.\n\n  Args:\n    input_graph_def: A GraphDef containing a model.\n    nodes_to_convert: A list of PlaceholderWithDefault or Placeholder nodes to\n      be converted to Constants with their new value.\n\n  Returns:\n    modified graph with PlaceholderWithDefault node converted to Constant node\n  '
    input_node_map = {}
    for node in input_graph_def.node:
        if node.name not in input_node_map:
            input_node_map[node.name] = node
        else:
            raise ValueError('Duplicate node names detected for ', node.name)
    dict_to_change = {}
    for key in PLACEHOLDER_WITH_DEFAULT_LIST:
        dict_to_change[key] = PLACEHOLDER_WITH_DEFAULT_LIST[key]
    if nodes_to_convert is not None and len(nodes_to_convert) > 0:
        dict_list = parse_nodes_dict(nodes_to_convert)
        dict_to_change.update(dict_list)
    ph_node_list = []
    for ph_node in dict_to_change:
        if not ph_node and ph_node not in input_node_map:
            continue
        ph_node_list.append(ph_node)
    if not ph_node_list:
        tf_logging.warning('No PlaceholderWithDefault nodes found to convert to Constant. Maybe check the spellings')
        return input_graph_def
    result_graph_def = graph_pb2.GraphDef()
    for node in input_graph_def.node:
        is_replaced = False
        new_node = node_def_pb2.NodeDef()
        if node.op == 'PlaceholderWithDefault' or node.op == 'Placeholder':
            match_key = [find_key for find_key in dict_to_change.keys() if find_key in node.name]
            if len(match_key) > 0:
                if dtypes.bool.as_datatype_enum == node.attr['dtype'].type:
                    new_val_str = dict_to_change[match_key[0]]
                    new_node.op = 'Const'
                    new_node.name = node.name
                    new_node.attr['dtype'].CopyFrom(node.attr['dtype'])
                    new_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(strtobool(new_val_str), dtype=dtypes.bool, shape=[])))
                    is_replaced = True
                else:
                    tf_logging.warning('Not converting to Const. Currently only bool             PlaceholderWithDefault or Placeholder can be converted to const.             current dtype = ', node.attr['dtype'])
        if not is_replaced:
            new_node.CopyFrom(node)
        result_graph_def.node.extend([new_node])
    return result_graph_def