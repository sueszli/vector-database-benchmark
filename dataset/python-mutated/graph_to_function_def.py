"""Utility to convert a Graph to a FunctionDef."""
import re
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry

def _make_argname_from_tensor_name(name):
    if False:
        return 10
    return re.sub(':0$', '', name).replace(':', '_o')

def _tensor_to_argdef(t, name=None, used_names=None):
    if False:
        while True:
            i = 10
    'Convert tensor t to an argdef, with a specified name or a unique name.'
    arg = op_def_pb2.OpDef.ArgDef()
    if name is None:
        arg.name = _make_argname_from_tensor_name(t.name)
        if used_names is not None:
            if arg.name in used_names:
                i = 0
                while True:
                    new_name = '%s_U%d' % (arg.name, i)
                    if new_name not in used_names:
                        arg.name = new_name
                        break
                    i += 1
            used_names.add(arg.name)
    else:
        arg.name = name
    arg.type = t.dtype.as_datatype_enum
    return arg

def _is_in_placeholders(op, func_arg_placeholders):
    if False:
        print('Hello World!')
    'Checks whether any output of this op is in func_arg_placeholders.'
    return op.values() and any((x.name in func_arg_placeholders for x in op.values()))

def _get_node_def(op):
    if False:
        print('Hello World!')
    return op.node_def

def _get_op_def(op):
    if False:
        for i in range(10):
            print('nop')
    return op.op_def or op_def_registry.get(op.type)

def _create_input_dict(function_graph, func_arg_placeholders, initial_value=None):
    if False:
        i = 10
        return i + 15
    'Create a mapping from graph tensor names to function tensor names.'
    if initial_value is None:
        input_dict = {}
    else:
        input_dict = dict(initial_value)
    for op in function_graph.get_operations():
        if _is_in_placeholders(op, func_arg_placeholders):
            input_dict[op.name] = op.name
        else:
            op_def = _get_op_def(op)
            attrs = _get_node_def(op).attr
            o = 0
            for arg_def in op_def.output_arg:
                if arg_def.number_attr:
                    num = attrs[arg_def.number_attr].i
                elif arg_def.type_list_attr:
                    num = len(attrs[arg_def.type_list_attr].list.type)
                else:
                    num = 1
                for i in range(num):
                    result = '%s:%s:%d' % (op.name, arg_def.name, i)
                    input_dict[op.values()[o].name] = result
                    if o == 0:
                        input_dict[op.name] = result
                    o += 1
    return input_dict

def _add_op_node(op, func, input_dict):
    if False:
        print('Hello World!')
    'Converts an op to a function def node and add it to `func`.'
    func.node_def.extend([_get_node_def(op)])
    node_def = func.node_def[-1]
    for i in range(len(node_def.input)):
        if not node_def.input[i].startswith('^'):
            assert node_def.input[i] in input_dict, '%s missing from %s' % (node_def.input[i], input_dict.items())
            node_def.input[i] = input_dict[node_def.input[i]]
    if op.op_def is not None and op.op_def.is_stateful:
        func.signature.is_stateful = True

def graph_to_function_def(graph, operations, inputs, outputs, out_names=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns `graph` as a `FunctionDef` protocol buffer.\n\n  This method creates a [`FunctionDef`](\n  https://www.tensorflow.org/code/tensorflow/core/framework/function.proto)\n  protocol buffer that contains all the ops in `operations`.  The\n  operations become the body of the function.\n\n  The arguments `inputs` and `outputs` will be listed as the inputs\n  and outputs tensors of the function.  They must be lists of\n  tensors present in the graph.  The lists can optionally be empty.\n\n  Args:\n    graph: Graph.\n    operations: the operations to put in the function. Must be a subset of\n     the operations in the graph.\n    inputs: List of tensors. Inputs to the function.\n    outputs: List of tensors. Outputs of the function.\n    out_names: Optional list of string names for the outputs.\n\n  Returns:\n    A FunctionDef protocol buffer.\n\n  Raises:\n    ValueError: if out_names is specified and the wrong length.\n  '
    func = function_pb2.FunctionDef()
    func.signature.name = '_'
    used_names = set()
    func.signature.input_arg.extend([_tensor_to_argdef(i, used_names=used_names) for i in inputs])
    initial_dict = {}
    for (o, m) in zip(inputs, func.signature.input_arg):
        initial_dict[o.name] = m.name
    if out_names is None:
        used_names = set()
        func.signature.output_arg.extend([_tensor_to_argdef(o, used_names=used_names) for o in outputs])
    elif len(outputs) != len(out_names):
        raise ValueError(f'out_names must be either empty or equal in size to outputs. len(out_names) = {len(out_names)} len(outputs) = {len(outputs)}')
    elif len(out_names) != len(set(out_names)):
        raise ValueError(f'Must not have duplicates in out_names. Received: {out_names}')
    else:
        func.signature.output_arg.extend([_tensor_to_argdef(o, name=n) for (o, n) in zip(outputs, out_names)])
    func_arg_placeholders = set((i.name for i in inputs))
    input_dict = _create_input_dict(graph, func_arg_placeholders, initial_value=initial_dict)
    for op in operations:
        if _is_in_placeholders(op, func_arg_placeholders):
            continue
        _add_op_node(op, func, input_dict)
    if out_names is None:
        for (index, o) in enumerate(outputs):
            k = func.signature.output_arg[index].name
            func.ret[k] = input_dict[o.name]
    else:
        for (o, n) in zip(outputs, out_names):
            func.ret[n] = input_dict[o.name]
    return func