"""TensorFlow Debugger: Tools for debugging gradients."""
import re
import uuid
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variables
_GRADIENT_DEBUG_TAG = 'gradient_debug_'
_gradient_debuggers = {}

def _tensor_to_grad_debug_op_name(tensor, grad_debugger_uuid):
    if False:
        return 10
    (op_name, slot) = debug_graphs.parse_node_or_tensor_name(tensor.name)
    return '%s_%d/%s%s' % (op_name, slot, _GRADIENT_DEBUG_TAG, grad_debugger_uuid)

def _parse_grad_debug_op_name(op_name):
    if False:
        while True:
            i = 10
    'Parse the name of a debug gradient op.\n\n  Args:\n    op_name: the name of the debug gradient op.\n\n  Returns:\n    1) The UUID of the GradientsDebugger that created the debug gradient op.\n    2) Name of the original tensor whose gradient is debugged by the debug\n       gradient op.\n  '
    name_items = op_name.split('/')
    assert len(name_items) > 1
    assert name_items[-1].startswith(_GRADIENT_DEBUG_TAG)
    grad_debugger_uuid = name_items[-1][len(_GRADIENT_DEBUG_TAG):]
    if '_' in grad_debugger_uuid:
        grad_debugger_uuid = grad_debugger_uuid[:grad_debugger_uuid.index('_')]
    orig_tensor_slot = int(name_items[-2][name_items[-2].rfind('_') + 1:])
    orig_base_op_name = name_items[-2][:name_items[-2].rfind('_')]
    orig_tensor_name = '/'.join(name_items[:-2] + [orig_base_op_name]) + ':%d' % orig_tensor_slot
    return (grad_debugger_uuid, orig_tensor_name)

class GradientsDebugger:
    """Gradients Debugger.

  Allows retrieval of gradient tensors created by TensorFlow's automatic
  differentiation algorithm, i.e., `tf.gradients` and optimizer classes that
  use it.
  """

    def __init__(self, y_tensor=None):
        if False:
            print('Hello World!')
        'Constructor of GradientsDebugger.\n\n    Args:\n      y_tensor: optional: the `tf.Tensor` to be differentiated, i.e., the tensor\n        on the numerator of the differentiation.\n    '
        self._uuid = uuid.uuid4().hex
        _gradient_debuggers[self._uuid] = self
        self._gradient_tensors = {}
        self._y_tensor = y_tensor
        self._graph = None
        if y_tensor:
            self._graph = y_tensor.graph
        self._is_active_context = False

    @property
    def y_tensor(self):
        if False:
            i = 10
            return i + 15
        return self._y_tensor

    @property
    def graph(self):
        if False:
            while True:
                i = 10
        return self._graph

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self._is_active_context = True

    def __exit__(self, unused_type, unused_value, unused_traceback):
        if False:
            print('Hello World!')
        self._is_active_context = False

    def identify_gradient(self, input_tensor):
        if False:
            return 10
        'Create a debug identity tensor that registers and forwards gradients.\n\n    The side effect of this method is that when gradient tensor(s) are created\n    with respect to the any paths that include the `input_tensor`, the gradient\n    tensor(s) with respect to `input_tensor` will be registered with this\n    this `GradientsDebugger` instance and can later be retrieved, with the\n    methods `gradient_tensor` and `gradient_tensors`.\n\n    Example:\n\n    ```python\n    x = tf.Variable(1.0)\n    y = tf.add(x, x)\n\n    grad_debugger = tf_debug.GradientsDebugger()\n    debug_y = grad_debugger.identify_gradient(y)\n    z = tf.square(debug_y)\n\n    # Create a train op under the grad_debugger context.\n    with grad_debugger:\n      train_op = tf.compat.v1.train.GradientDescentOptimizer(z)\n\n    # Now we can reflect through grad_debugger to get the gradient tensor\n    # with respect to y.\n    y_grad = grad_debugger.gradient_tensor(y)\n    ```\n\n    Args:\n      input_tensor: the input `tf.Tensor` object whose related gradient tensors\n        are to be registered with this `GradientsDebugger` instance when they\n        are created, e.g., during `tf.gradients` calls or the construction\n        of optimization (training) op that uses `tf.gradients`.\n\n    Returns:\n      A forwarded identity of `input_tensor`, as a `tf.Tensor`.\n\n    Raises:\n      ValueError: If an op with name that duplicates the gradient-debugging op\n        already exists in the graph (highly unlikely).\n    '
        grad_debug_op_name = _tensor_to_grad_debug_op_name(input_tensor, self._uuid)
        identity_op = gen_array_ops.debug_gradient_ref_identity if input_tensor.dtype._is_ref_dtype else gen_array_ops.debug_gradient_identity
        debug_grad_identity = identity_op(input_tensor, name=grad_debug_op_name)
        assert debug_grad_identity.dtype == input_tensor.dtype
        if debug_grad_identity.op.name != grad_debug_op_name:
            raise ValueError('The graph already contains an op named %s' % grad_debug_op_name)
        return debug_grad_identity

    def watch_gradients_by_tensors(self, graph, tensors):
        if False:
            print('Hello World!')
        'Watch gradient tensors by x-tensor(s).\n\n    The side effect of this method is that when gradient tensor(s) are created\n    with respect to the any paths that include the `x_tensor`s, the gradient\n    tensor(s) with respect to the tensor will be registered with this\n    this `GradientsDebugger` instance and can later be retrieved, with the\n    methods `gradient_tensor` and `gradient_tensors`.\n\n    Unlike the method `identify_gradient`, this method is used to retrieve\n    gradient tensors after the construction of the forward subgraph has\n    completed (but before the construction of the backward subgraph).\n\n    This method is the same as `watch_gradients_by_x_tensor_names` except that\n    the tensors are specified by the Python `tf.Tensor` or `tf.Variable`\n    objects, instead by name patterns.\n\n    Example:\n\n    ```python\n    x = tf.Variable(1.0)\n    y = tf.add(x, x, name="y")\n    z = tf.square(debug_y)\n\n    # Create a train op under the grad_debugger context.\n    grad_debugger = tf_debug.GradientsDebugger()\n    with grad_debugger.watch_gradients_by_tensors(y):\n      train_op = tf.compat.v1.train.GradientDescentOptimizer(z)\n\n    # Now we can reflect through grad_debugger to get the gradient tensor\n    # with respect to y.\n    y_grad = grad_debugger.gradient_tensor(y)\n    # or\n    y_grad = grad_debugger.gradient_tensor("y:0")\n    ```\n\n    Args:\n      graph: the `tf.Graph` to watch the gradients on.\n      tensors: a `tf.Tensor` or `tf.Variable` object, or a list of such objects.\n\n    Returns:\n      The GradientsDebugger instance itself.\n    '
        if not isinstance(tensors, list):
            tensors = [tensors]
        tensor_name_regex = []
        for tensor in tensors:
            tensor_name_regex.append(re.escape(tensor.name) + '$')
        tensor_name_regex = '(' + '|'.join(tensor_name_regex) + ')'
        return self.watch_gradients_by_tensor_names(graph, tensor_name_regex)

    def watch_gradients_by_tensor_names(self, graph, tensor_name_regex):
        if False:
            return 10
        'Watch gradient tensors by name(s) of the x-tensor(s).\n\n    The side effect of this method is that when gradient tensor(s) are created\n    with respect to the x-tensors, the gradient tensor(s) will be registered\n    with this `GradientsDebugger` instance and can later be retrieved.\n\n    Unlike the `identify_gradient` method, this method is used after the\n    construction of the forward graph has completed. Unlike the\n    `watch_gradients_by_tensor` method, this method does not use handles to the\n    tensors of interest; it uses their names.\n\n    This method is the same as `watch_gradients_by_tensors` except that the\n    x-tensors are specified by name patterns, instead of `tf.Tensor` or\n    `tf.Variable` objects.\n\n    Example:\n\n    ```python\n    x = tf.Variable(1.0, name="x")\n    y = tf.add(x, x, name="y")\n    z = tf.square(debug_y)\n\n    # Create a train op under the grad_debugger context.\n    grad_debugger = tf_debug.GradientsDebugger()\n    with grad_debugger.watch_gradients_by_tensor_names(r"(x|y):0$"):\n      train_op = tf.compat.v1.train.GradientDescentOptimizer(z)\n\n    # Now we can reflect through grad_debugger to get the gradient tensor\n    # with respect to x and y.\n    x_grad = grad_debugger.gradient_tensor("x:0")\n    y_grad = grad_debugger.gradient_tensor("y:0")\n    ```\n\n    Args:\n      graph: the `tf.Graph` to watch the gradients on.\n      tensor_name_regex: the regular-expression pattern of the name(s) of the\n        x-tensor(s) to watch. x-tensor refers to the tensors on the denominator\n        of the differentiation.\n\n    Returns:\n      The GradientsDebugger instance itself.\n    '
        tensor_name_pattern = re.compile(tensor_name_regex)
        with graph.as_default():
            for op in graph.get_operations():
                for output in op.outputs:
                    if tensor_name_pattern.match(output.name):
                        debug_op = self.identify_gradient(output)
                        for consumer in list(output.consumers()):
                            if consumer == debug_op.op:
                                continue
                            for (i, consumer_input) in enumerate(consumer.inputs):
                                if consumer_input == output:
                                    consumer._update_input(i, debug_op)
        return self

    def _check_same_graph(self, tensor):
        if False:
            print('Hello World!')
        if self._graph is None:
            self._graph = tensor.graph
        elif self._graph != tensor.graph:
            raise ValueError('The graph of the value (%s) is not the same as the graph %s' % (tensor.graph, self._graph))

    def register_gradient_tensor(self, x_tensor_name, gradient_tensor):
        if False:
            print('Hello World!')
        'Register the gradient tensor for an x-tensor.\n\n    Args:\n      x_tensor_name: (`str`) the name of the independent `tf.Tensor`, i.e.,\n        the tensor on the denominator of the differentiation.\n      gradient_tensor: the gradient `tf.Tensor`.\n    '
        if len(_gradient_debuggers) == 1 or self._is_active_context:
            self._check_same_graph(gradient_tensor)
            self._gradient_tensors[x_tensor_name] = gradient_tensor

    def gradient_tensor(self, x_tensor):
        if False:
            for i in range(10):
                print('nop')
        'Get the gradient tensor of an x-tensor.\n\n    Args:\n      x_tensor: (`tf.Tensor`, `tf.Variable` or `str`) The x-tensor object or its\n        name. x-tensor refers to the independent `tf.Tensor`, i.e., the tensor\n        on the denominator of the differentiation.\n\n    Returns:\n      If found, the gradient tensor.\n\n    Raises:\n      TypeError: If `x_tensor` is not a `tf.Tensor`, `tf.Variable` or `str`.\n      LookupError: If the `x_tensor` has not been registered with a gradient\n        tensor.\n    '
        x_tensor_name = self._get_tensor_name(x_tensor)
        if x_tensor_name not in self._gradient_tensors:
            raise LookupError('This GradientsDebugger has not received any gradient tensor for x-tensor %s' % x_tensor_name)
        return self._gradient_tensors[x_tensor_name]

    def gradient_tensors(self):
        if False:
            while True:
                i = 10
        'Get the gradient tensors that this object is aware of.\n\n    Returns:\n      A dict mapping x-tensor names to gradient tensor objects. x-tensor refers\n      to the tensors on the denominator of the differentation.\n    '
        return self._gradient_tensors

    def _get_tensor_name(self, tensor):
        if False:
            while True:
                i = 10
        if isinstance(tensor, (tensor_lib.Tensor, variables.Variable)):
            return tensor.name
        elif isinstance(tensor, str):
            return tensor
        else:
            raise TypeError('x_tensor must be a str or tf.Tensor or tf.Variable, but instead has type %s' % type(tensor))

def clear_gradient_debuggers():
    if False:
        for i in range(10):
            print('nop')
    'Clear all globally registered gradient debuggers.'
    _gradient_debuggers.clear()

@ops.RegisterGradient('DebugGradientIdentity')
def _identify_gradient_grad(op, dy):
    if False:
        i = 10
        return i + 15
    'Gradient function for the DebugIdentity op.'
    (grad_debugger_uuid, orig_tensor_name) = _parse_grad_debug_op_name(op.name)
    grad_debugger = _gradient_debuggers[grad_debugger_uuid]
    grad_debugger.register_gradient_tensor(orig_tensor_name, dy)
    return dy

@ops.RegisterGradient('DebugGradientRefIdentity')
def _identify_gradient_grad_ref(op, dy):
    if False:
        return 10
    'Gradient function for the DebugIdentity op.'
    return _identify_gradient_grad(op, dy)

def gradient_values_from_dump(grad_debugger, x_tensor, dump):
    if False:
        print('Hello World!')
    'Find gradient values from a `DebugDumpDir` object.\n\n  Args:\n    grad_debugger: the `tf_debug.GradientsDebugger` instance to be used.\n    x_tensor: (`tf.Tensor`, `tf.Variable` or `str`) The x-tensor object or its\n      name. x-tensor refers to the independent `tf.Tensor`, i.e., the tensor\n      on the denominator of the differentiation.\n    dump: A `tfdbg.DebugDumpDir` object.\n\n  Returns:\n    If this `GradientsDebugger` instance has the gradient tensor of `x_tensor`\n      registered: a list of `numpy.ndarray` representing the value of the\n      gradient tensor from `dump`. The list could be empty, if the gradient\n      tensor is not executed in the `tf.Session.run()` call that generated\n      the `dump`. The list could also contain multiple values of the gradient\n      tensor, e.g., if gradient tensor is computed repeatedly in a\n      `tf.while_loop` during the run that generated the `dump`.\n\n  Raises:\n    LookupError: If this `GradientsDebugger` instance does not have the\n      gradient tensor of `x_tensor` registered.\n    ValueError: If this `GradientsDebugger` has a `tf.Graph` object that\n      does not match the `tf.Graph` object of the `dump`.\n    TypeError: If `x_tensor` is not a `tf.Tensor`, `tf.Variable` or `str`.\n  '
    if dump.python_graph and grad_debugger.graph and (dump.python_graph != grad_debugger.graph):
        raise ValueError('This GradientsDebugger instance has a graph (%s) that differs from the graph of the DebugDumpDir object (%s).' % (grad_debugger.graph, dump.python_graph))
    gradient_tensor = grad_debugger.gradient_tensor(x_tensor)
    (node_name, output_slot) = debug_graphs.parse_node_or_tensor_name(gradient_tensor.name)
    try:
        return dump.get_tensors(node_name, output_slot, 'DebugIdentity')
    except debug_data.WatchKeyDoesNotExistInDebugDumpDirError:
        return []