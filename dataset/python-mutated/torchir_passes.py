from collections import OrderedDict
import logging as _logging
from .internal_graph import *

def transform_inplace_ops(graph, name_remap_dict=None):
    if False:
        return 10
    if name_remap_dict is None:
        name_remap_dict = {}
    for node in graph.nodes:
        for (k, v) in name_remap_dict.items():
            node.replace_name(k, v)
        if node.kind == 'append':
            if isinstance(node.parent, InternalTorchIRGraph):
                name_remap_dict[node.inputs[0]] = node.outputs[0]
            elif node.parent.parent.kind == 'loop':
                global_input = node.inputs[0]
                local_input = node.parent.parent.name + '.0'
                local_output = node.outputs[0]
                global_output = local_output + '.out'
                name_remap_dict[global_input] = global_output
                node.parent.parent.inputs.append(global_input)
                node.parent.inputs.append(local_input)
                node.replace_name(global_input, local_input)
                node.parent.outputs.append(local_output)
                node.parent.parent.outputs.append(global_output)
                node.parent.parent.name = node.parent.parent.outputs[0]
            elif node.parent.parent.kind == 'if':
                raise NotImplementedError("inplace_ops pass doesn't yet support append op inside conditional")
        for block in node.blocks:
            transform_inplace_ops(block, name_remap_dict)
    for (k, v) in name_remap_dict.items():
        try:
            idx = graph.outputs.index(k)
        except ValueError:
            pass
        else:
            graph.outputs[idx] = v

def flatten_graph_input_values(graph):
    if False:
        print('Hello World!')
    " CoreML can't handle nested iterables of tensors, so we flatten the\n        inputs of any graph that expects them.\n    "
    new_graph_inputs = graph.inputs
    all_new_nodes = []
    changed = True
    notified = False
    while changed:
        old_graph_inputs = new_graph_inputs
        new_graph_inputs = OrderedDict()
        new_nodes = []
        changed = False
        for (_input_name, _input_val) in old_graph_inputs.items():
            if isinstance(_input_val, (tuple, list)):
                changed = True
                if not notified:
                    notified = True
                    _logging.warning('Tuple detected at graph input. This will be flattened in the converted model.')
                node_inputs = []
                for (idx, item) in enumerate(_input_val):
                    name = _input_name + '_{}'.format(idx)
                    new_graph_inputs[name] = item
                    node_inputs.append(name)
                new_nodes.append(InternalTorchIRNode(inputs=node_inputs, outputs=[_input_name], kind='tupleconstruct'))
            else:
                new_graph_inputs[_input_name] = _input_val
        all_new_nodes = new_nodes + all_new_nodes
    graph.inputs = new_graph_inputs
    graph.nodes = all_new_nodes + graph.nodes

def flatten_graph_output_values(graph):
    if False:
        print('Hello World!')
    " CoreML can't handle nested iterables of tensors, so we flatten the\n        outputs of any graph that produces them.\n    "
    node_names = [node.name for node in graph.nodes]
    new_graph_outputs = graph.outputs
    changed = True
    notified = False
    while changed:
        old_graph_outputs = new_graph_outputs
        new_graph_outputs = []
        changed = False
        for outp in old_graph_outputs:
            try:
                node_idx = node_names.index(outp)
            except:
                new_graph_outputs.append(outp)
                continue
            if graph.nodes[node_idx].kind in ['tupleconstruct', 'listconstruct']:
                new_graph_outputs.extend(graph.nodes[node_idx].inputs)
                changed = True
                if not notified:
                    notified = True
                    _logging.warning('Tuple detected at graph output. This will be flattened in the converted model.')
            else:
                new_graph_outputs.append(outp)
    graph.outputs = new_graph_outputs