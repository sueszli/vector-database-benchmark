"""Utility to lift subgraphs."""
import collections
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import op_selector
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
UnliftableError = op_selector.UnliftableError

def _as_operation(op_or_tensor):
    if False:
        i = 10
        return i + 15
    if isinstance(op_or_tensor, tensor_lib.Tensor):
        return op_or_tensor.op
    return op_or_tensor

def _constant_inputs(op_or_tensor):
    if False:
        return 10
    return all((_as_operation(i).type == u'Const' and (not _as_operation(i).control_inputs) for i in op_selector.graph_inputs(_as_operation(op_or_tensor))))
_InputMutation = collections.namedtuple('_InputMutation', ['copied_op', 'input_index', 'old_graph_tensor'])
_ControlMutation = collections.namedtuple('_ControlMutation', ['copied_op', 'old_graph_op'])

def _copy_non_source(op, graph, op_map, base_graph):
    if False:
        return 10
    "Copy an op directly to a given graph.\n\n  Generally `op`'s inputs should already have been copied. If this is not the\n  case, for example with v1 while_loops, then `_copy_non_source` inserts\n  placeholders for the unavailable Tensors and returns a list of required\n  mutations.\n\n  Args:\n    op: The op to be copied.\n    graph: The destination graph.\n    op_map: A dict mapping ops and tensors in the old graph to the new one.\n    base_graph: The graph we're copying from, for any necessary functions.\n  Returns:\n    A tuple of (required_inputs, required_control_inputs):\n      required_inputs:\n        A list of `_InputMutation` tuples containing inputs to `copied_op` which\n        must be updated once `old_graph_tensor` has been copied.\n      required_control_inputs:\n        A list of `_ControlMutation` tuples containing control inputs to\n        `copied_op` which must be added once `old_graph_op` has been copied.\n  "
    input_mutations = []
    control_mutations = []
    copied_inputs = []
    for (input_index, original_input) in enumerate(op.inputs):
        copied_input = op_map.get(original_input, None)
        if copied_input is None:
            copied_input = array_ops.placeholder(name='unused_control_flow_input', shape=original_input.shape, dtype=original_input.dtype)
            input_mutations.append(_InputMutation(copied_op=None, input_index=input_index, old_graph_tensor=original_input))
        copied_inputs.append(copied_input)
    copied_control_inputs = []
    for original_control_input in op.control_inputs:
        copied_control_input = op_map.get(original_control_input, None)
        if copied_control_input is None:
            control_mutations.append(_ControlMutation(copied_op=None, old_graph_op=original_control_input))
        else:
            copied_control_inputs.append(copied_control_input)
    with ops.control_dependencies(copied_control_inputs), ops.device(op.device):
        f = base_graph._functions.get(op.type, None)
        if f is not None and compat.as_str(f.name) not in graph._functions:
            f.add_to_graph(graph)
        copied_op = graph.create_op(op_type=op.type, inputs=copied_inputs, dtypes=[x.dtype for x in op.outputs], attrs={key: value for (key, value) in op.node_def.attr.items() if not key.startswith('_class') and (not key.startswith('_tpu_replicate'))}, name=op.name)
    op_map[op] = copied_op
    for (i, o) in enumerate(op.outputs):
        op_map[o] = copied_op.outputs[i]
    return ([mutation._replace(copied_op=copied_op) for mutation in input_mutations], [mutation._replace(copied_op=copied_op) for mutation in control_mutations])

def _copy_source(s, graph, op_map, handle_captures, inverse_captures, base_graph):
    if False:
        while True:
            i = 10
    'Create a source in a graph based on a Tensor from a different graph.\n\n  This function creates a placeholder analog of `s` in a graph with the\n  following behavior:\n\n  1) If s is a captured Tensor or Variable and handle_captures is set to True,\n     simply capture it in the new graph as well.\n\n  2) If s is a PlaceholderWithDefault whose default is a constant, preserve\n     said default in the new graph.\n\n  3) When applicable, copy resource variable metadata from `s` to the newly\n     created placeholder.\n\n  Args:\n    s: The source of interest.\n    graph: The destination graph.\n    op_map: A dict mapping ops and tensors in the old graph to the new one.\n    handle_captures: A boolean indicating whether to re-capture s in the new\n      graph or simply create a vanilla placeholder.\n    inverse_captures: A dict mapping s back to the Tensor or Variable that it\n      captures.\n    base_graph: The graph being copied from.\n  '
    if handle_captures and s in inverse_captures:
        copied_placeholder = graph.capture(inverse_captures[s], name=s.op.name)
    elif s.op.type == 'PlaceholderWithDefault' and _constant_inputs(s):
        default_value = s.op.inputs[0]
        (unavailable_inputs, unavailable_control_inputs) = _copy_non_source(op=default_value.op, graph=graph, op_map=op_map, base_graph=base_graph)
        if unavailable_inputs or unavailable_control_inputs:
            raise AssertionError('Could not copy source node {} because it has inputs.'.format(default_value))
        with ops.device(s.op.device):
            copied_placeholder = array_ops.placeholder_with_default(input=op_map[default_value], shape=s.shape, name=s.op.name)
    else:
        with ops.device(s.op.device):
            copied_placeholder = array_ops.placeholder(dtype=s.dtype, shape=s.shape, name=s.op.name)
    base_handle = resource_variable_ops.get_resource_handle_data(s)
    if base_handle.shape_and_type:
        resource_variable_ops._set_handle_shapes_and_types(copied_placeholder, base_handle, graph_mode=True)
    op_map[s] = copied_placeholder
    op_map[s.op] = copied_placeholder.op

@tf_export('__internal__.lift_to_graph', v1=[])
def lift_to_graph(tensors, graph, sources=None, disallowed_placeholders=None, add_sources=False, handle_captures=False, base_graph=None, op_map=None):
    if False:
        return 10
    "Copies the tensor and all its inputs recursively to the outer graph.\n\n  Args:\n    tensors: The Tensors to lift.\n    graph: The graph to lift to.\n    sources: Optional sequence of nodes to start from. If omitted the whole\n      subgraph which feeds into `init_tensor` is lifted.\n    disallowed_placeholders: An optional set of ops which may not appear in the\n      lifted graph. Defaults to all placeholders.\n    add_sources: A boolean indicating whether placeholders which are not in\n      sources should be allowed.\n    handle_captures: A boolean indicating whether to re-capture s in the new\n      graph or simply create a vanilla placeholder.\n    base_graph: The graph from which to lift ops. This will be inferred if not\n      specified.\n    op_map: A map contains all the existing nodes that have been lifted to the\n      destination graph, so they won't be lifted and copied again.\n\n  Returns:\n    A mapping from ops in the current default graph to ops in `graph`.\n\n  Raises:\n    UnliftableError: If a placeholder blocks lifting.\n  "
    variable_init_tensors = []
    init_tensors = []
    for tensor in tensors:
        if isinstance(tensor, resource_variable_ops.ResourceVariable):
            variable_init_tensors.append(tensor)
        else:
            init_tensors.append(tensor)
    base_graph = base_graph or init_tensors[0].graph
    op_map = op_map or object_identity.ObjectIdentityDictionary()
    sources = object_identity.ObjectIdentitySet(sources or [])
    visited_ops = set((x.op for x in sources))
    op_outputs = collections.defaultdict(set)
    for init_tensor in init_tensors:
        sources.update(op_selector.map_subgraph(init_tensor=init_tensor, sources=sources, disallowed_placeholders=disallowed_placeholders, visited_ops=visited_ops, op_outputs=op_outputs, add_sources=add_sources))
    ops_to_copy = []
    marked_ops = set([])
    ops_to_visit = [_as_operation(t) for t in init_tensors if not op_outputs[_as_operation(t)]]
    unvisited_ops = set(ops_to_visit)
    while unvisited_ops:
        while ops_to_visit:
            op = ops_to_visit.pop()
            if op in marked_ops:
                continue
            marked_ops.add(op)
            ops_to_copy.append(op)
            for inp in op_selector.graph_inputs(op):
                if inp.type == 'TPUReplicateMetadata':
                    continue
                unvisited_ops.add(inp)
                if all((x in marked_ops for x in op_outputs[inp])) and inp not in sources:
                    ops_to_visit.append(inp)
        unvisited_ops.difference_update(marked_ops)
        if unvisited_ops:
            ops_to_visit.append(next(iter(unvisited_ops)))
    ops_to_copy.sort(key=lambda op: len(op_selector.graph_inputs(op)) == 0)
    captures = []
    inverse_captures = object_identity.ObjectIdentityDictionary()
    internal_captures = []
    if isinstance(base_graph, func_graph.FuncGraph) and isinstance(graph, func_graph.FuncGraph):
        captures = base_graph.captures
        for (external_capture, internal_capture) in captures:
            inverse_captures[internal_capture] = external_capture
        internal_captures = base_graph.internal_captures
    with graph.as_default():
        for i in variable_init_tensors:
            op_map[i] = i
        source_ops = set()
        for s in internal_captures:
            if s in sources:
                sources.remove(s)
                source_ops.add(s.op)
                _copy_source(s=s, graph=graph, op_map=op_map, handle_captures=handle_captures, inverse_captures=inverse_captures, base_graph=base_graph)
        for s in sources:
            source_ops.add(s.op)
            _copy_source(s=s, graph=graph, op_map=op_map, handle_captures=handle_captures, inverse_captures=inverse_captures, base_graph=base_graph)
        input_mutations = []
        control_mutations = []
        for op in reversed(ops_to_copy):
            if op in source_ops or op in op_map:
                continue
            (new_input_mutations, new_control_mutations) = _copy_non_source(op=op, graph=graph, op_map=op_map, base_graph=base_graph)
            input_mutations.extend(new_input_mutations)
            control_mutations.extend(new_control_mutations)
        with graph._mutation_lock():
            for mutation in input_mutations:
                mutation.copied_op._update_input(mutation.input_index, op_map[mutation.old_graph_tensor])
            for mutation in control_mutations:
                if mutation.old_graph_op.type == 'TPUReplicateMetadata':
                    continue
                mutation.copied_op._add_control_input(op_map[mutation.old_graph_op])
        return op_map