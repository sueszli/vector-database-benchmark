from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from ..parsed_tf_node import ParsedTFNode
import copy

def insert_get_tuple(gddict):
    if False:
        for i in range(10):
            print('nop')
    '\n    TensorFlow uses input "nodename:i" to denote "get tuple i" from "nodename".\n    Here we split it so that:\n\n    node1:i -> node2\n\n    gets transformed into\n\n    node1 -> get_tuple(i) --> node2\n\n    Takes a graph in "dict{str, ParsedTFNode}" form, and returns a new graph.\n\n    We do not do this for control flow nodes(Switch, Enter, Exit, Merge\n    LoopCond, NextIteration).  For these nodes, we just convert\n\n    node1:i -> node2\n\n    to\n\n    node1 -> node2\n    '
    retdict = {}
    get_tuple_op_var_index = 1
    inserted_ops = {}

    def make_op(input_node, index, new_node_name, gto_make_op_cache):
        if False:
            while True:
                i = 10
        cache_key = (input_node, index)
        if cache_key in gto_make_op_cache:
            return gto_make_op_cache[cache_key]
        inserted_op_name = new_node_name
        inserted_op = ParsedTFNode()
        inserted_op.name = inserted_op_name
        inserted_op.op = 'get_tuple'
        inserted_op.inputs = [input_node]
        inserted_op.attr['index'] = index
        inserted_ops[inserted_op_name] = inserted_op
        gto_make_op_cache[cache_key] = inserted_op
        return inserted_op
    exclusions = ['Switch', 'Enter', 'Exit', 'Merge', 'LoopCond', 'NextIteration', 'TensorArrayV3', 'Const']
    inclusions = ['Split', 'SplitV', 'LSTMBlockCell']
    gto_make_op_cache = {}
    for name in list(gddict.keys()):
        new_node = ParsedTFNode()
        new_node = copy.deepcopy(gddict[name])
        new_inputs = []
        for idx in range(len(new_node.inputs)):
            if ':' in new_node.inputs[idx]:
                (input_node, input_index) = new_node.inputs[idx].split(':')
            else:
                input_node = new_node.inputs[idx]
                input_index = 0
            if '_output_shapes' in gddict[input_node].attr and len(gddict[input_node].attr['_output_shapes']) > 1 and (gddict[input_node].op not in exclusions) or gddict[input_node].op in inclusions:
                get_tuple_node_name = 'gto_%s' % get_tuple_op_var_index
                new_inputs.append(make_op(input_node, int(input_index), get_tuple_node_name, gto_make_op_cache).name)
                get_tuple_op_var_index += 1
            else:
                new_inputs.append(new_node.inputs[idx])
        new_node.inputs = new_inputs
        retdict[name] = new_node
    for (k, v) in inserted_ops.items():
        retdict[k] = v
    for (k, v) in retdict.items():
        for idx in range(len(v.inputs)):
            if ':' in v.inputs[idx]:
                (nodename, nodeindex) = v.inputs[idx].split(':')
                v.inputs[idx] = nodename
    return retdict