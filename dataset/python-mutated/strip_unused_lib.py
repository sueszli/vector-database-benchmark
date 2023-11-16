"""Utilities to remove unneeded nodes from a GraphDefs."""
import copy
from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

def strip_unused(input_graph_def, input_node_names, output_node_names, placeholder_type_enum):
    if False:
        i = 10
        return i + 15
    'Removes unused nodes from a GraphDef.\n\n  Args:\n    input_graph_def: A graph with nodes we want to prune.\n    input_node_names: A list of the nodes we use as inputs.\n    output_node_names: A list of the output nodes.\n    placeholder_type_enum: The AttrValue enum for the placeholder data type, or\n        a list that specifies one value per input node name.\n\n  Returns:\n    A `GraphDef` with all unnecessary ops removed.\n\n  Raises:\n    ValueError: If any element in `input_node_names` refers to a tensor instead\n      of an operation.\n    KeyError: If any element in `input_node_names` is not found in the graph.\n  '
    for name in input_node_names:
        if ':' in name:
            raise ValueError(f"Name '{name}' appears to refer to a Tensor, not an Operation.")
    not_found = {name for name in input_node_names}
    inputs_replaced_graph_def = graph_pb2.GraphDef()
    for node in input_graph_def.node:
        if node.name in input_node_names:
            not_found.remove(node.name)
            placeholder_node = node_def_pb2.NodeDef()
            placeholder_node.op = 'Placeholder'
            placeholder_node.name = node.name
            if isinstance(placeholder_type_enum, list):
                input_node_index = input_node_names.index(node.name)
                placeholder_node.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=placeholder_type_enum[input_node_index]))
            else:
                placeholder_node.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=placeholder_type_enum))
            if '_output_shapes' in node.attr:
                placeholder_node.attr['_output_shapes'].CopyFrom(node.attr['_output_shapes'])
            if 'shape' in node.attr:
                placeholder_node.attr['shape'].CopyFrom(node.attr['shape'])
            inputs_replaced_graph_def.node.extend([placeholder_node])
        else:
            inputs_replaced_graph_def.node.extend([copy.deepcopy(node)])
    if not_found:
        raise KeyError(f'The following input nodes were not found: {not_found}.')
    output_graph_def = graph_util.extract_sub_graph(inputs_replaced_graph_def, output_node_names)
    return output_graph_def

def strip_unused_from_files(input_graph, input_binary, output_graph, output_binary, input_node_names, output_node_names, placeholder_type_enum):
    if False:
        while True:
            i = 10
    'Removes unused nodes from a graph file.'
    if not gfile.Exists(input_graph):
        print("Input graph file '" + input_graph + "' does not exist!")
        return -1
    if not output_node_names:
        print('You need to supply the name of a node to --output_node_names.')
        return -1
    input_graph_def = graph_pb2.GraphDef()
    mode = 'rb' if input_binary else 'r'
    with gfile.GFile(input_graph, mode) as f:
        if input_binary:
            input_graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), input_graph_def)
    output_graph_def = strip_unused(input_graph_def, input_node_names.split(','), output_node_names.split(','), placeholder_type_enum)
    if output_binary:
        with gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
    else:
        with gfile.GFile(output_graph, 'w') as f:
            f.write(text_format.MessageToString(output_graph_def))
    print('%d ops in the final graph.' % len(output_graph_def.node))