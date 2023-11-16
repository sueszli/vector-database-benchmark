from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

def tensor_array_resource_removal(gd):
    if False:
        while True:
            i = 10
    for (k, node) in gd.items():
        if node.op.startswith('TensorArray') and node.op != 'TensorArrayV3':
            node.inputs = node.inputs[1:]
        for i in range(len(node.inputs)):
            if ':' in node.inputs[i]:
                (input_node, input_index) = node.inputs[i].split(':')
                input_index = int(input_index)
            else:
                input_node = node.inputs[i]
                input_index = 0
            if gd[input_node].op == 'TensorArrayV3':
                if input_index == 1:
                    node.inputs[i] = '%s' % input_node