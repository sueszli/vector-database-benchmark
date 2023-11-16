"""Subscribe function."""
import contextlib
import re
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging

def _recursive_apply(tensors, apply_fn):
    if False:
        i = 10
        return i + 15
    'Helper method to recursively apply a function to structure of tensors.\n\n  The structure of the tensors should take the form similar to fetches in\n  `tf.compat.v1.Session` and includes single `Tensor`, `list`, nested `list`,\n  `tuple`,\n  `namedtuple`, or `dict`.\n\n  Args:\n    tensors: Single `Tensor`, `list`, nested `list, `tuple`, `namedtuple`, or\n      `dict`.\n    apply_fn: Function to apply to each `Tensor` and should return a `Tensor`.\n\n  Returns:\n    Returns the modified tensors with the same structure.\n  Raises:\n    `TypeError` if undefined type in the tensors structure.\n  '
    tensors_type = type(tensors)
    if isinstance(tensors, tensor_lib.Tensor):
        return apply_fn(tensors)
    elif isinstance(tensors, variables.Variable):
        return apply_fn(tensors.value())
    elif isinstance(tensors, (list, tuple)):
        tensors = [_recursive_apply(t, apply_fn) for t in tensors]
        if tensors_type is list:
            return list(tensors)
        elif tensors_type is tuple:
            return tuple(tensors)
        return tensors_type(*tensors)
    elif tensors_type is dict:
        return dict(((k, _recursive_apply(v, apply_fn)) for (k, v) in tensors.items()))
    else:
        raise TypeError(f'_recursive_apply argument {tensors!r} has invalid type {tensors_type!r}')

class _ControlOutputCache(object):
    """Helper class to manage calculating and caching control_outputs in graph."""
    __slots__ = ['cache']

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.cache = {}

    def calc_control_outputs(self, graph):
        if False:
            i = 10
            return i + 15
        'Returns the map of control_outputs for a given graph.\n\n    Args:\n      graph: The graph to parse.\n\n    Returns:\n      A map of the control outputs.\n    '
        control_outputs = {}
        for op in graph.get_operations():
            for control_input in op.control_inputs:
                if control_input not in control_outputs:
                    control_outputs[control_input] = set()
                control_outputs[control_input].add(op)
        return control_outputs

    def get_control_outputs(self, op):
        if False:
            return 10
        'Return the control outputs for a given op.\n\n    Args:\n      op: The op to fetch control outputs for.\n\n    Returns:\n      Iterable of control output ops.\n    '
        if op.graph not in self.cache:
            control_outputs = self.calc_control_outputs(op.graph)
            self.cache[op.graph] = control_outputs
        else:
            control_outputs = self.cache[op.graph]
        return control_outputs.get(op, [])

def _subscribe_new(tensor, side_effects, control_cache):
    if False:
        return 10
    'Helper method that subscribes a single tensor to a list of side_effects.\n\n  Args:\n    tensor: `tf.Tensor`\n    side_effects: List of side_effect functions see subscribe for details.\n    control_cache: `_ControlOutputCache` helper to get control_outputs faster.\n\n  Returns:\n    The modified replacement to the passed in tensor which triggers the side\n    effects.\n  '
    update_input = []
    for consumer_op in list(tensor.consumers()):
        update_input.append((consumer_op, list(consumer_op.inputs).index(tensor)))
    update_control_input = control_cache.get_control_outputs(tensor.op)
    name_scope = tensor.op.name + '/subscription/'
    with ops.name_scope(name_scope):
        outs = []
        for s in side_effects:
            outs += s(tensor)
        with ops.control_dependencies(outs):
            out = array_ops.identity(tensor)
    for (consumer_op, index) in update_input:
        consumer_op._update_input(index, out)
    for consumer_op in update_control_input:
        new_control_inputs = consumer_op.control_inputs
        if tensor.op in new_control_inputs:
            new_control_inputs.remove(tensor.op)
        new_control_inputs.append(out.op)
        consumer_op._remove_all_control_inputs()
        consumer_op._add_control_inputs(new_control_inputs)
    return out

def _subscribe_extend(tensor, side_effects):
    if False:
        print('Hello World!')
    'Helper method to extend the list of side_effects for a subscribed tensor.\n\n  Args:\n    tensor: A `tf.Tensor` as returned by subscribe().\n    side_effects: List of side_effect functions, see subscribe for details.\n\n  Returns:\n    The given subscribed tensor (for API consistency).\n  '
    assert len(tensor.op.inputs) == 1, 'Op {} must only have one input'.format(tensor.op.name)
    source_tensor = tensor.op.inputs[0]
    outs = []
    name_scope = source_tensor.op.name + '/subscription/'
    with ops.name_scope(name_scope):
        for s in side_effects:
            outs += s(source_tensor)
    out_ops = [out.op if isinstance(out, tensor_lib.Tensor) else out for out in outs]
    tensor.op._add_control_inputs(out_ops)
    return tensor

def _is_subscribed_identity(tensor):
    if False:
        for i in range(10):
            print('nop')
    'Checks if the given tensor is an identity op returned by `subscribe()`.\n\n  Args:\n    tensor: A `tf.Tensor` to check.\n\n  Returns:\n    True if the given tensor matches the criteria for subscription identities:\n    its op type is `Identity`, its name matches the name of its input and\n    conforms to the convention for subscribed nodes.\n    False otherwise.\n  '
    if tensor.op.type != 'Identity':
        return False
    match = re.match('(?P<prefix_name>^.*?)/subscription/Identity[^/]+', tensor.name)
    if match is None or len(match.groups()) != 1:
        return False
    prefix_name = match.group('prefix_name')
    assert len(tensor.op.inputs) == 1, 'Op {} must only have one input'.format(tensor.op.name)
    source_tensor = tensor.op.inputs[0]
    if prefix_name != source_tensor.op.name:
        return False
    return True

def _subscribe(tensor, side_effects, control_cache):
    if False:
        i = 10
        return i + 15
    "Helper method that subscribes a single tensor to a list of side_effects.\n\n  This method will check if the given tensor has already been subscribed or if\n  it's a tensor returned by a previous call to `subscribe()` and, if so, will\n  reuse the existing identity op, appending the given side effects to the list\n  of existing ones.\n\n  Args:\n    tensor: The `tf.Tensor` to be subscribed.\n    side_effects: List of side_effect functions, see subscribe for details.\n    control_cache: `_ControlOutputCache` helper to get control_outputs faster.\n\n  Returns:\n    The modified replacement to the passed in tensor which triggers the side\n    effects or the given tensor, if it was already been subscribed.\n  "
    if not tensor.dtype.is_numpy_compatible:
        logging.debug('Tensor {} has an un-supported {} type and cannot be subscribed.'.format(tensor.name, tensor.dtype))
        return tensor
    if _is_subscribed_identity(tensor):
        return _subscribe_extend(tensor, side_effects)
    name_scope = tensor.op.name + '/subscription/Identity'
    consumers = tensor.consumers()
    matching_ops = [op for op in consumers if op.name.startswith(name_scope)]
    assert len(matching_ops) <= 1, 'Op {} must only have one subscription op connected to it'.format(tensor.op.name)
    if len(matching_ops) == 1:
        candidate_tensor = matching_ops[0].outputs[0]
        if _is_subscribed_identity(candidate_tensor):
            return _subscribe_extend(candidate_tensor, side_effects)
    return _subscribe_new(tensor, side_effects, control_cache)

@contextlib.contextmanager
def _preserve_control_flow_context(tensor):
    if False:
        print('Hello World!')
    "Preserve the control flow context for the given tensor.\n\n  Sets the graph context to the tensor's context so that side effect ops are\n  added under the same context.\n\n  This is needed when subscribing to tensors defined within a conditional\n  block or a while loop. In these cases we need that the side-effect ops\n  are created within the same control flow context as that of the tensor\n  they are attached to.\n\n  Args:\n    tensor: tensor whose context should be preserved.\n\n  Yields:\n    None\n  "
    context = tensor.op._get_control_flow_context()
    if context:
        context.Enter()
    try:
        yield
    finally:
        if context:
            context.Exit()

def _scoped_subscribe(tensor, side_effects, control_cache):
    if False:
        print('Hello World!')
    'Helper method that subscribes a single tensor to a list of side_effects.\n\n  This is a thin wrapper around `_subscribe` and ensures that the side effect\n  ops are added within the same device and control flow context of the\n  subscribed tensor.\n\n  Args:\n    tensor: The `tf.Tensor` to be subscribed.\n    side_effects: List of side_effect functions, see subscribe for details.\n    control_cache: `_ControlOutputCache` helper to get control_outputs faster.\n\n  Returns:\n    The modified replacement to the passed in tensor which triggers the side\n    effects or the given tensor, if it was already been subscribed.\n  '
    with ops.device(tensor.device):
        with _preserve_control_flow_context(tensor):
            return _subscribe(tensor, side_effects, control_cache)

def subscribe(tensors, side_effects):
    if False:
        i = 10
        return i + 15
    "Subscribe to a tensor.\n\n  This method will attach side effect graphs to a given set\n  of tensors. Set of tensors follows from session.run and supports\n  single `Tensor`, `list`, nested `list`, `tuple`, `namedtuple`, or `dict`. It\n  returns the tensors in the same passed in structure, but as clones with\n  side effects applied. The supplied side effect graphs are specified\n  as a constructor function which takes the target tensor and\n  constructs a side effect graph and returns a list of ops that should\n  be control dependencies on fetching the tensor. It will append\n  'subscription' to the name scope of the tensor for every node in\n  the side effect graph. These control dependencies are what trigger\n  the side effects. Subscribe will construct the additions to your\n  graph and return the created identity tensor downstream of the control\n  dependencies. Use these tensors as you would normally in the rest of\n  your tensorflow code. If a given tensor has already been subscribed or a\n  tensor returned by a call to subscribe is passed, the previously created\n  identity tensor will be reused and the side effect graphs will be added to\n  the existing ones.\n\n  Args:\n    tensors: `Tensor` or set of tensors to subscribe to. Set of tensors format\n      follows from `Session.run` and supports single `Tensor`, `list`, nested\n      `list`, `tuple`, `namedtuple`, or `dict`.\n    side_effects: Function(s) that takes a `Tensor`, construct a subgraph, and\n      return a nonempty list of control dependencies. This can be a single\n      function or list of functions.\n\n  Returns:\n    Subscribed tensors, which are identity copies of the passed in tensors\n      in the same passed in structure, but the graph has been modified\n      such that these are downstream of the control dependencies for\n      the side effect graphs. Use these functionally equivalent tensors\n      instead of the passed in tensors for further construction or running.\n  "
    if not hasattr(side_effects, '__iter__'):
        side_effects = [side_effects]
    control_outputs = _ControlOutputCache()
    result = _recursive_apply(tensors, lambda t: _scoped_subscribe(t, side_effects, control_outputs))
    return result