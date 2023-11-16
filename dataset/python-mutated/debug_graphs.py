"""Classes and methods for processing debugger-decorated graphs."""
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.platform import tf_logging as logging

def parse_node_or_tensor_name(name):
    if False:
        print('Hello World!')
    'Get the node name from a string that can be node or tensor name.\n\n  Args:\n    name: An input node name (e.g., "node_a") or tensor name (e.g.,\n      "node_a:0"), as a str.\n\n  Returns:\n    1) The node name, as a str. If the input name is a tensor name, i.e.,\n      consists of a colon, the final colon and the following output slot\n      will be stripped.\n    2) If the input name is a tensor name, the output slot, as an int. If\n      the input name is not a tensor name, None.\n  '
    if ':' in name and (not name.endswith(':')):
        node_name = name[:name.rfind(':')]
        output_slot = int(name[name.rfind(':') + 1:])
        return (node_name, output_slot)
    else:
        return (name, None)

def get_node_name(element_name):
    if False:
        for i in range(10):
            print('nop')
    (node_name, _) = parse_node_or_tensor_name(element_name)
    return node_name

def get_output_slot(element_name):
    if False:
        while True:
            i = 10
    'Get the output slot number from the name of a graph element.\n\n  If element_name is a node name without output slot at the end, 0 will be\n  assumed.\n\n  Args:\n    element_name: (`str`) name of the graph element in question.\n\n  Returns:\n    (`int`) output slot number.\n  '
    (_, output_slot) = parse_node_or_tensor_name(element_name)
    return output_slot if output_slot is not None else 0

def is_copy_node(node_name):
    if False:
        for i in range(10):
            print('nop')
    'Determine whether a node name is that of a debug Copy node.\n\n  Such nodes are inserted by TensorFlow core upon request in\n  RunOptions.debug_options.debug_tensor_watch_opts.\n\n  Args:\n    node_name: Name of the node.\n\n  Returns:\n    A bool indicating whether the input argument is the name of a debug Copy\n    node.\n  '
    return node_name.startswith('__copy_')

def is_debug_node(node_name):
    if False:
        print('Hello World!')
    'Determine whether a node name is that of a debug node.\n\n  Such nodes are inserted by TensorFlow core upon request in\n  RunOptions.debug_options.debug_tensor_watch_opts.\n\n  Args:\n    node_name: Name of the node.\n\n  Returns:\n    A bool indicating whether the input argument is the name of a debug node.\n  '
    return node_name.startswith('__dbg_')

def parse_debug_node_name(node_name):
    if False:
        for i in range(10):
            print('nop')
    'Parse the name of a debug node.\n\n  Args:\n    node_name: Name of the debug node.\n\n  Returns:\n    1. Name of the watched node, as a str.\n    2. Output slot index of the watched tensor, as an int.\n    3. Index of the debug node, as an int.\n    4. Name of the debug op, as a str, e.g, "DebugIdentity".\n\n  Raises:\n    ValueError: If the input node name is not a valid debug node name.\n  '
    prefix = '__dbg_'
    name = node_name
    if not name.startswith(prefix):
        raise ValueError("Invalid prefix in debug node name: '%s'" % node_name)
    name = name[len(prefix):]
    if name.count('_') < 2:
        raise ValueError("Invalid debug node name: '%s'" % node_name)
    debug_op = name[name.rindex('_') + 1:]
    name = name[:name.rindex('_')]
    debug_op_index = int(name[name.rindex('_') + 1:])
    name = name[:name.rindex('_')]
    if name.count(':') != 1:
        raise ValueError("Invalid tensor name in debug node name: '%s'" % node_name)
    watched_node_name = name[:name.index(':')]
    watched_output_slot = int(name[name.index(':') + 1:])
    return (watched_node_name, watched_output_slot, debug_op_index, debug_op)

class GraphTracingReachedDestination(Exception):
    pass

class DFSGraphTracer(object):
    """Graph input tracer using depth-first search."""

    def __init__(self, input_lists, skip_node_names=None, destination_node_name=None):
        if False:
            for i in range(10):
                print('nop')
        'Constructor of _DFSGraphTracer.\n\n    Args:\n      input_lists: A list of dicts. Each dict is an adjacency (input) map from\n        the recipient node name as the key and the list of input node names\n        as the value.\n      skip_node_names: Optional: a list of node names to skip tracing.\n      destination_node_name: Optional: destination node name. If not `None`, it\n        should be the name of a destination not as a str and the graph tracing\n        will raise GraphTracingReachedDestination as soon as the node has been\n        reached.\n\n    Raises:\n      GraphTracingReachedDestination: if stop_at_node_name is not None and\n        the specified node is reached.\n    '
        self._input_lists = input_lists
        self._skip_node_names = skip_node_names
        self._inputs = []
        self._visited_nodes = []
        self._depth_count = 0
        self._depth_list = []
        self._destination_node_name = destination_node_name

    def trace(self, graph_element_name):
        if False:
            i = 10
            return i + 15
        'Trace inputs.\n\n    Args:\n      graph_element_name: Name of the node or an output tensor of the node, as a\n        str.\n\n    Raises:\n      GraphTracingReachedDestination: if destination_node_name of this tracer\n        object is not None and the specified node is reached.\n    '
        self._depth_count += 1
        node_name = get_node_name(graph_element_name)
        if node_name == self._destination_node_name:
            raise GraphTracingReachedDestination()
        if node_name in self._skip_node_names:
            return
        if node_name in self._visited_nodes:
            return
        self._visited_nodes.append(node_name)
        for input_list in self._input_lists:
            if node_name not in input_list:
                continue
            for inp in input_list[node_name]:
                if get_node_name(inp) in self._visited_nodes:
                    continue
                self._inputs.append(inp)
                self._depth_list.append(self._depth_count)
                self.trace(inp)
        self._depth_count -= 1

    def inputs(self):
        if False:
            while True:
                i = 10
        return self._inputs

    def depth_list(self):
        if False:
            for i in range(10):
                print('nop')
        return self._depth_list

def _infer_device_name(graph_def):
    if False:
        for i in range(10):
            print('nop')
    'Infer device name from a partition GraphDef.'
    device_name = None
    for node in graph_def.node:
        if node.device:
            device_name = node.device
            break
    if device_name is None:
        logging.warn('Failed to infer device name from partition GraphDef: none of the nodes of the GraphDef has a non-empty device name.')
    return device_name

class DebugGraph(object):
    """Represents a debugger-decorated graph."""

    def __init__(self, debug_graph_def, device_name=None):
        if False:
            while True:
                i = 10
        self._debug_graph_def = debug_graph_def
        self._non_debug_graph_def = None
        self._node_attributes = {}
        self._node_inputs = {}
        self._node_reversed_ref_inputs = {}
        self._node_ctrl_inputs = {}
        self._node_recipients = {}
        self._node_ctrl_recipients = {}
        self._node_devices = {}
        self._node_op_types = {}
        self._copy_send_nodes = []
        self._ref_args = {}
        self._device_name = device_name
        if not self._device_name:
            self._device_name = _infer_device_name(debug_graph_def)
        for node in debug_graph_def.node:
            self._process_debug_graph_node(node)
        self._prune_non_control_edges_of_debug_ops()
        self._prune_control_edges_of_debug_ops()
        self._prune_nodes_from_input_and_recipient_maps(self._get_copy_nodes())
        self._populate_recipient_maps()

    def _process_debug_graph_node(self, node):
        if False:
            print('Hello World!')
        'Process a node from the debug GraphDef.\n\n    Args:\n      node: (NodeDef) A partition-graph node to be processed.\n\n    Raises:\n      ValueError: If duplicate node names are encountered.\n    '
        if is_debug_node(node.name):
            return
        if node.name in self._node_inputs:
            raise ValueError("Duplicate node name on device %s: '%s'" % (self._device_name, node.name))
        self._node_attributes[node.name] = node.attr
        self._node_inputs[node.name] = []
        self._node_ctrl_inputs[node.name] = []
        self._node_recipients[node.name] = []
        self._node_ctrl_recipients[node.name] = []
        if node.name not in self._node_devices:
            self._node_devices[node.name] = set()
        self._node_devices[node.name].add(node.device if node.device else self._device_name)
        self._node_op_types[node.name] = node.op
        self._ref_args[node.name] = self._get_ref_args(node)
        for inp in node.input:
            if is_copy_node(inp) and (node.op == '_Send' or node.op == '_Retval'):
                self._copy_send_nodes.append(node.name)
            if inp.startswith('^'):
                cinp = inp[1:]
                self._node_ctrl_inputs[node.name].append(cinp)
            else:
                self._node_inputs[node.name].append(inp)

    def _get_ref_args(self, node):
        if False:
            i = 10
            return i + 15
        'Determine whether an input of an op is ref-type.\n\n    Args:\n      node: A `NodeDef`.\n\n    Returns:\n      A list of the arg names (as strs) that are ref-type.\n    '
        op_def = op_def_registry.get(node.op)
        if op_def is None:
            return []
        ref_args = []
        for (i, output_arg) in enumerate(op_def.output_arg):
            if output_arg.is_ref:
                arg_name = node.name if i == 0 else '%s:%d' % (node.name, i)
                ref_args.append(arg_name)
        return ref_args

    def _get_copy_nodes(self):
        if False:
            i = 10
            return i + 15
        'Find all Copy nodes in the loaded graph.'
        copy_nodes = []
        for node in self._node_inputs:
            if is_copy_node(node):
                copy_nodes.append(node)
        return copy_nodes

    def _prune_non_control_edges_of_debug_ops(self):
        if False:
            for i in range(10):
                print('nop')
        'Prune (non-control) edges related to debug ops.\n\n    Prune the Copy ops and associated _Send ops inserted by the debugger out\n    from the non-control inputs and output recipients map. Replace the inputs\n    and recipients with original ones.\n    '
        for node in self._node_inputs:
            inputs = self._node_inputs[node]
            for (i, inp) in enumerate(inputs):
                if is_copy_node(inp):
                    orig_inp = self._node_inputs[inp][0]
                    inputs[i] = orig_inp

    def _prune_control_edges_of_debug_ops(self):
        if False:
            while True:
                i = 10
        'Prune control edges related to the debug ops.'
        for node in self._node_ctrl_inputs:
            ctrl_inputs = self._node_ctrl_inputs[node]
            debug_op_inputs = []
            for ctrl_inp in ctrl_inputs:
                if is_debug_node(ctrl_inp):
                    debug_op_inputs.append(ctrl_inp)
            for debug_op_inp in debug_op_inputs:
                ctrl_inputs.remove(debug_op_inp)

    def _populate_recipient_maps(self):
        if False:
            i = 10
            return i + 15
        'Populate the map from node name to recipient(s) of its output(s).\n\n    This method also populates the input map based on reversed ref edges.\n    '
        for node in self._node_inputs:
            inputs = self._node_inputs[node]
            for inp in inputs:
                inp = get_node_name(inp)
                if inp not in self._node_recipients:
                    self._node_recipients[inp] = []
                self._node_recipients[inp].append(node)
                if inp in self._ref_args:
                    if inp not in self._node_reversed_ref_inputs:
                        self._node_reversed_ref_inputs[inp] = []
                    self._node_reversed_ref_inputs[inp].append(node)
        for node in self._node_ctrl_inputs:
            ctrl_inputs = self._node_ctrl_inputs[node]
            for ctrl_inp in ctrl_inputs:
                if ctrl_inp in self._copy_send_nodes:
                    continue
                if ctrl_inp not in self._node_ctrl_recipients:
                    self._node_ctrl_recipients[ctrl_inp] = []
                self._node_ctrl_recipients[ctrl_inp].append(node)

    def _prune_nodes_from_input_and_recipient_maps(self, nodes_to_prune):
        if False:
            i = 10
            return i + 15
        'Prune nodes out of input and recipient maps.\n\n    Args:\n      nodes_to_prune: (`list` of `str`) Names of the nodes to be pruned.\n    '
        for node in nodes_to_prune:
            del self._node_inputs[node]
            del self._node_ctrl_inputs[node]
            del self._node_recipients[node]
            del self._node_ctrl_recipients[node]

    def _reconstruct_non_debug_graph_def(self):
        if False:
            i = 10
            return i + 15
        'Reconstruct non-debug GraphDef.\n\n    Non-debug GraphDef means the original GraphDef without the Copy* and Debug\n    nodes inserted by the debugger.\n    '
        if self._non_debug_graph_def:
            return
        self._non_debug_graph_def = graph_pb2.GraphDef()
        for node in self._debug_graph_def.node:
            if is_copy_node(node.name) or is_debug_node(node.name):
                continue
            new_node = self._non_debug_graph_def.node.add()
            new_node.CopyFrom(node)
            del new_node.input[:]
            for inp in self._node_inputs[node.name]:
                new_node.input.append(inp)
            for ctrl_inp in self._node_ctrl_inputs[node.name]:
                new_node.input.append('^' + ctrl_inp)

    @property
    def device_name(self):
        if False:
            print('Hello World!')
        return self._device_name

    @property
    def debug_graph_def(self):
        if False:
            print('Hello World!')
        'The debugger-decorated GraphDef.'
        return self._debug_graph_def

    @property
    def non_debug_graph_def(self):
        if False:
            print('Hello World!')
        'The GraphDef without the Copy* and Debug* nodes added by the debugger.'
        self._reconstruct_non_debug_graph_def()
        return self._non_debug_graph_def

    @property
    def node_devices(self):
        if False:
            i = 10
            return i + 15
        return self._node_devices

    @property
    def node_op_types(self):
        if False:
            i = 10
            return i + 15
        return self._node_op_types

    @property
    def node_attributes(self):
        if False:
            i = 10
            return i + 15
        return self._node_attributes

    @property
    def node_inputs(self):
        if False:
            print('Hello World!')
        return self._node_inputs

    @property
    def node_ctrl_inputs(self):
        if False:
            return 10
        return self._node_ctrl_inputs

    @property
    def node_reversed_ref_inputs(self):
        if False:
            return 10
        return self._node_reversed_ref_inputs

    @property
    def node_recipients(self):
        if False:
            print('Hello World!')
        return self._node_recipients

    @property
    def node_ctrl_recipients(self):
        if False:
            while True:
                i = 10
        return self._node_ctrl_recipients

def reconstruct_non_debug_graph_def(debug_graph_def):
    if False:
        while True:
            i = 10
    'Reconstruct original (non-debugger-decorated) partition GraphDef.\n\n  This method strips the input `tf.compat.v1.GraphDef` of the Copy* and\n  Debug*-type nodes inserted by the debugger.\n\n  The reconstructed partition graph is identical to the original (i.e.,\n    non-debugger-decorated) partition graph except in the following respects:\n      1) The exact names of the runtime-inserted internal nodes may differ.\n         These include _Send, _Recv, _HostSend, _HostRecv, _Retval ops.\n      2) As a consequence of 1, the nodes that receive input directly from such\n         send- and recv-type ops will have different input names.\n      3) The parallel_iteration attribute of while-loop Enter ops are set to 1.\n\n  Args:\n    debug_graph_def: The debugger-decorated `tf.compat.v1.GraphDef`, with the\n      debugger-inserted Copy* and Debug* nodes.\n\n  Returns:\n    The reconstructed `tf.compat.v1.GraphDef` stripped of the debugger-inserted\n    nodes.\n  '
    return DebugGraph(debug_graph_def).non_debug_graph_def