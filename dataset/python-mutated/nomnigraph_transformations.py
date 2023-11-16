from collections import defaultdict
import caffe2.python.nomnigraph as ng
from caffe2.python import core, utils

def transpose_network(nn):
    if False:
        i = 10
        return i + 15
    '\n    Convert all Convolutions operators which are in the NCHW order\n    to NHWC order and also transform their inputs and outputs so that the\n    rest of the graph is not affected.\n    '
    incoming = {}
    outgoing = defaultdict(lambda : [])
    dfg = nn.dataFlow
    orig_nodes = [x for x in nn.nodes]
    for node in orig_nodes:
        if node.isOperator() and node.name == 'Conv':
            arg_dict = utils.ArgsToDict(node.annotation.operator_def.arg)
            if 'order' in arg_dict and arg_dict['order'] != 'NCHW':
                continue
            inputs = [x for x in node.inputs]
            assert len(inputs) >= 2, 'Conv operator should have two inputs'
            outputs = [x for x in node.outputs]
            assert len(outputs) >= 1, 'Conv operator should have an output'
            for inp in inputs:
                nn.deleteEdge(inp, node)
            for outp in outputs:
                nn.deleteEdge(node, outp)
            for idx in range(2):
                new_inp = nn.createUniqueDataNode(inputs[idx].name)
                transp = dfg.createNode(ng.NeuralNetOperator('NCHW2NHWC'))
                nn.createEdge(inputs[idx], transp)
                nn.createEdge(transp, new_inp)
                outgoing[inputs[idx]].append(transp)
                inputs[idx] = new_inp
            for idx in range(len(outputs)):
                new_outp = nn.createUniqueDataNode(outputs[idx].name)
                transp = dfg.createNode(ng.NeuralNetOperator('NHWC2NCHW'))
                nn.createEdge(transp, outputs[idx])
                nn.createEdge(new_outp, transp)
                incoming[outputs[idx]] = new_outp
                outputs[idx] = new_outp
            arg_dict['order'] = 'NHWC'
            new_node = nn.createNode(core.CreateOperator('Conv', [], [], **arg_dict))
            for inp in inputs:
                nn.createEdge(inp, new_node)
            for outp in outputs:
                nn.createEdge(new_node, outp)
            nn.deleteNode(node)
    for orig_tensor in outgoing:
        if orig_tensor in incoming:
            new_tensor = incoming[orig_tensor]
        else:
            out_ops = outgoing[orig_tensor]
            new_tensor = out_ops[0].outputs[0]
            outgoing[orig_tensor] = out_ops[1:]
        for opnode in outgoing[orig_tensor]:
            for out in opnode.outputs:
                nn.replaceAllUsesWith(out, new_tensor)
                nn.deleteNode(out)
            nn.deleteNode(opnode)