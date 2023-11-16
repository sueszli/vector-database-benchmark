from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from ..basic_graph_ops import disconnect_vertex_ins, delete_node

def remove_variable_node_impl(fn, tfssa):
    if False:
        print('Hello World!')
    variables = [var for var in fn.graph.values() if var.op == 'VariableV2']
    assigns = [assign for assign in fn.graph.values() if assign.op == 'Assign']
    reads = [read for read in fn.graph.values() if read.op == 'Identity' and len(read.inputs) == 1 and (fn.graph[read.inputs[0]].op == 'VariableV2')]
    variable_values = {}
    additional_nodes_to_delete = []
    for v in variables:
        v.parse_from_attr()
        variable_values[v.name] = v.datatype()
        for node in fn.graph.values():
            if node.op == 'Assign' and node.inputs[0] == v.name and (node.inputs[1] == v.name + '/initial_value'):
                variable_values[v.name] = fn.graph[node.inputs[1]].value
                additional_nodes_to_delete += [node.name, node.inputs[1]]
    for r in reads:
        r.op = 'get_global'
        r.attr['variable'] = r.inputs[0]
        disconnect_vertex_ins(fn.graph, r.name)
    for r in assigns:
        r.op = 'set_global'
        r.attr['variable'] = r.inputs[0]
    for var in variables:
        delete_node(fn.graph, var.name)
    for node in additional_nodes_to_delete:
        delete_node(fn.graph, node)
    for (k, v) in variable_values.items():
        tfssa.variables[k] = v

def remove_variable_nodes(tfssa):
    if False:
        print('Hello World!')
    '\n    This should be performed after constant propagation pass.\n    '
    for v in tfssa.functions.values():
        remove_variable_node_impl(v, tfssa)