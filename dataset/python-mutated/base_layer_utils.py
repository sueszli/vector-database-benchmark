"""Contains private utilities used mainly by the base Layer class."""
import functools
import threading
from tensorflow.python import tf2
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.trackable import base as tracking
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
_call_context = threading.local()

def create_mean_metric(value, name=None):
    if False:
        while True:
            i = 10
    from tensorflow.python.keras import metrics as metrics_module
    metric_obj = metrics_module.Mean(name=name, dtype=value.dtype)
    return (metric_obj, metric_obj(value))

def make_variable(name, shape=None, dtype=dtypes.float32, initializer=None, trainable=None, caching_device=None, validate_shape=True, constraint=None, use_resource=None, collections=None, synchronization=tf_variables.VariableSynchronization.AUTO, aggregation=tf_variables.VariableAggregation.NONE, partitioner=None):
    if False:
        while True:
            i = 10
    'Temporary util to create a variable (relies on `variable_scope.variable`).\n\n  Some reuse-related technicalities prevent us from using\n  `variable_scope.get_variable()` directly, so we use a subcomponent\n  that has fewer constraints (`variable_scope.variable()`).\n\n  In the longer term, it seems like a similar "default variable creator" method\n  should exist in `Trackable` instead. When this happens, we can get\n  rid of this temporary solution.\n\n  TODO(fchollet): remove this method when no longer needed.\n\n  Args:\n    name: Variable name.\n    shape: Variable shape.\n    dtype: The type of the variable. Defaults to `self.dtype` or `float32`.\n    initializer: Initializer instance (callable).\n    trainable: Whether the variable should be part of the layer\'s\n      "trainable_variables" (e.g. variables, biases)\n      or "non_trainable_variables" (e.g. BatchNorm mean, stddev).\n      Note, if the current variable scope is marked as non-trainable\n      then this parameter is ignored and any added variables are also\n      marked as non-trainable. `trainable` defaults to `True` unless\n      `synchronization` is set to `ON_READ`.\n    caching_device: Passed to `tf.Variable`.\n    validate_shape: Passed to `tf.Variable`.\n    constraint: Constraint instance (callable).\n    use_resource: Whether to use a `ResourceVariable`.\n    collections: List of graph collections keys. The new variable is added to\n      these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.\n    synchronization: Indicates when a distributed a variable will be\n      aggregated. Accepted values are constants defined in the class\n      `tf.VariableSynchronization`. By default the synchronization is set to\n      `AUTO` and the current `DistributionStrategy` chooses\n      when to synchronize. If `synchronization` is set to `ON_READ`,\n      `trainable` must not be set to `True`.\n    aggregation: Indicates how a distributed variable will be aggregated.\n      Accepted values are constants defined in the class\n      `tf.VariableAggregation`.\n    partitioner: Not handled at this time.\n\n  Returns:\n    Variable instance.\n  '
    initializing_from_value = False
    if initializer is not None and (not callable(initializer)):
        initializing_from_value = True
    if initializing_from_value:
        init_val = initializer
        variable_dtype = None
    else:
        if tf_inspect.isclass(initializer):
            initializer = initializer()
        init_val = functools.partial(initializer, shape, dtype=dtype)
        variable_dtype = dtype.base_dtype
    if use_resource is None:
        use_resource = True
    variable_shape = tensor_shape.TensorShape(shape)
    return variable_v1.VariableV1(initial_value=init_val, name=name, trainable=trainable, caching_device=caching_device, dtype=variable_dtype, validate_shape=validate_shape, constraint=constraint, use_resource=use_resource, collections=collections, synchronization=synchronization, aggregation=aggregation, shape=variable_shape if variable_shape else None)

def collect_previous_mask(input_tensors):
    if False:
        return 10
    'Retrieves the output mask(s) of the previous node.\n\n  Args:\n      input_tensors: An arbitrary structure of Tensors.\n\n  Returns:\n      A mask tensor or list of mask tensors.\n  '

    def _collect_previous_mask(x):
        if False:
            return 10
        return getattr(x, '_keras_mask', None)
    return nest.map_structure(_collect_previous_mask, input_tensors)

def have_all_keras_metadata(tensors):
    if False:
        return 10
    return all((hasattr(x, '_keras_history') for x in nest.flatten(tensors)))

def generate_placeholders_from_shape(shape):
    if False:
        print('Hello World!')
    return array_ops.placeholder(shape=shape, dtype=backend.floatx())

def create_keras_history(tensors):
    if False:
        i = 10
        return i + 15
    'Wraps TensorFlow Operations for compatibility with the Functional API.\n\n  This method checks to see if a Tensor in `tensors` is missing Keras metadata\n  and has its origin in a Keras `Input` Layer. If so, this method will replace\n  the raw TensorFlow Operations that created this tensor with\n  `TensorFlowOpLayer` instances that create identical operations.\n\n  Any Tensors not originating from a Keras `Input` Layer will be treated as\n  constants when constructing `TensorFlowOpLayer` instances.\n\n  Args:\n    tensors: A structure of Tensors, some of which come from raw TensorFlow\n      operations and need to have Keras metadata assigned to them.\n\n  Returns:\n    created_layers: List. The `TensorFlowOpLayer` instances created to wrap\n      the raw Tensorflow operations.\n  '
    (_, created_layers) = _create_keras_history_helper(tensors, set(), [])
    return created_layers
_UNSAFE_GRAPH_OP_LAYER_CREATION = False

def _create_keras_history_helper(tensors, processed_ops, created_layers):
    if False:
        for i in range(10):
            print('nop')
    'Helper method for `create_keras_history`.\n\n  Args:\n    tensors: A structure of Tensors for which to create Keras metadata.\n    processed_ops: Set. TensorFlow operations that have already been wrapped in\n      `TensorFlowOpLayer` instances.\n    created_layers: List. The `TensorFlowOpLayer` instances created.\n\n  Returns:\n    Tuple. First element is the updated set of TensorFlow Operations that\n    have been wrapped in `TensorFlowOpLayer` instances. Second element is\n    a list of the `TensorFlowOpLayer` instances created.\n  '
    if ops.executing_eagerly_outside_functions():
        raise ValueError('`create_keras_history` should only be called if eager is disabled!')
    from tensorflow.python.keras.engine import base_layer
    tensor_list = nest.flatten(tensors)
    sparse_ops = []
    ragged_tensors = []
    for tensor in tensor_list:
        if getattr(tensor, '_keras_history', None) is not None:
            continue
        if isinstance(tensor, (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
            sparse_ops.append(tensor.op)
            continue
        if tf_utils.is_ragged(tensor):
            ragged_tensors.append(tensor)
            continue
        op = tensor.op
        if op not in processed_ops:
            op_inputs = list(op.inputs)
            constants = {}
            layer_inputs = []
            for (i, op_input) in enumerate(op_inputs):
                if uses_keras_history(op_input):
                    layer_inputs.append(op_input)
                else:
                    ds_with_session = distribute_lib.in_cross_replica_context() and (not ops.executing_eagerly_outside_functions())
                    using_xla = control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph())
                    if ds_with_session or using_xla or _UNSAFE_GRAPH_OP_LAYER_CREATION:
                        constants[i] = op_input
                    else:
                        with ops.init_scope():
                            constants[i] = backend.function([], op_input)([])
            layer_inputs = unnest_if_single_tensor(layer_inputs)
            (processed_ops, created_layers) = _create_keras_history_helper(layer_inputs, processed_ops, created_layers)
            name = op.name
            node_def = op.node_def.SerializeToString()
            op_layer = base_layer.TensorFlowOpLayer(node_def, constants=constants, name=name)
            created_layers.append(op_layer)
            op_layer._set_connectivity_metadata(args=(layer_inputs,), kwargs={}, outputs=op.outputs)
            processed_ops.update([op])
    if sparse_ops or ragged_tensors:
        lambda_example = '\n    weights_mult = lambda x: tf.sparse.sparse_dense_matmul(x, weights)\n    output = tf.keras.layers.Lambda(weights_mult)(input)\n    '
        raise ValueError('Tensorflow ops that generate ragged or sparse tensor outputs are currently not supported by Keras automatic op wrapping. Please wrap these ops in a Lambda layer: \n\n```\n{example}\n```\nSparse ops encountered: {sparse_ops}\nRagged tensors encountered: {ragged_tensors}\n'.format(example=lambda_example, sparse_ops=str(sparse_ops), ragged_tensors=str(ragged_tensors)))
    return (processed_ops, created_layers)

def unnest_if_single_tensor(input_tensors):
    if False:
        i = 10
        return i + 15
    flat_input_tensors = nest.flatten(input_tensors)
    if not isinstance(input_tensors, dict) and len(flat_input_tensors) == 1:
        input_tensors = flat_input_tensors[0]
    return input_tensors

def needs_keras_history(tensors, ignore_call_context=False):
    if False:
        print('Hello World!')
    'Check if any Tensors need to be wrapped in TensorFlowOpLayers.\n\n  This will never return True inside a sublayer, because sublayers\n  do not need to create Keras History. Otherwise, this returns True\n  if one or more of `tensors` originates from a `keras.Input` and\n  does not have `_keras_history` set.\n\n  Args:\n    tensors: An arbitrary nested structure of Tensors.\n    ignore_call_context: Whether to ignore the check of if currently\n      outside of a `call` context. This is `True` when creating\n      KerasHistory inside `Node`, where we always know that Tensors\n      are being used with the Functional API.\n\n  Returns:\n    Bool, whether at least one Tensor needs to be wrapped.\n  '
    input_tensors = nest.flatten(tensors)
    if call_context().in_call and (not ignore_call_context):
        return False
    if all((getattr(tensor, '_keras_history', None) is not None for tensor in input_tensors)):
        return False
    return uses_keras_history(tensors)

def is_in_keras_graph():
    if False:
        print('Hello World!')
    'Returns if currently executing inside of a Keras graph.'
    return call_context().in_keras_graph

def is_in_eager_or_tf_function():
    if False:
        while True:
            i = 10
    'Returns if in eager mode or inside of a tf.function.'
    return context.executing_eagerly() or is_in_tf_function()

def is_in_tf_function():
    if False:
        print('Hello World!')
    'Returns if inside of a tf.function.'
    if not ops.executing_eagerly_outside_functions():
        return False
    if not ops.inside_function():
        return False
    if is_in_keras_graph():
        return False
    graph = ops.get_default_graph()
    if getattr(graph, 'name', False) and graph.name.startswith('wrapped_function'):
        return False
    return True

def uses_keras_history(tensors):
    if False:
        return 10
    'Check if at least one Tensor originates from a `keras.Input`.\n\n  This is `True` if at least one Tensor has its origin in a `keras.Input`.\n  Any Tensor that originates from a `keras.Input` will have a dependency\n  Tensor with a `_keras_history` attribute attached. Tensors that have\n  already been checked to not originate from a `keras.Input`\n  are marked as `_keras_history_checked`.\n\n  Args:\n    tensors: An arbitrary nested structure of Tensors.\n\n  Returns:\n    Bool, whether at least one Tensor originates from a `keras.Input`.\n  '
    checked_tensors = set()
    tensors_to_check = nest.flatten(tensors)
    while tensors_to_check:
        new_tensors_to_check = []
        for tensor in tensors_to_check:
            if id(tensor) in checked_tensors:
                continue
            checked_tensors.add(id(tensor))
            if getattr(tensor, '_keras_history_checked', None) is not None:
                continue
            if getattr(tensor, '_keras_history', None) is not None:
                return True
            try:
                new_tensors_to_check.extend(tensor.op.inputs)
            except AttributeError:
                pass
        tensors_to_check = new_tensors_to_check
    mark_checked(tensors)
    return False

def mark_checked(tensors):
    if False:
        for i in range(10):
            print('nop')
    'Marks that these Tensors should not be tracked.\n\n  This prevents Layers from attempting to create TensorFlowOpLayers\n  for these Tensors.\n\n  Args:\n    tensors: An arbitrary structure of Tensors.\n  '

    def _mark_checked(tensor):
        if False:
            while True:
                i = 10
        tensor._keras_history_checked = True
    nest.map_structure(_mark_checked, tensors)

def call_context():
    if False:
        i = 10
        return i + 15
    'Returns currently active `CallContext`.'
    call_ctx = getattr(_call_context, 'call_context', None)
    if call_ctx is None:
        call_ctx = CallContext()
        _call_context.call_context = call_ctx
    return call_ctx

class CallContext(object):
    """Keeps track of properties currently inside a Layer/Model's `call`.

  Attributes:
    in_call: Whether currently inside the `call` of a Layer.
    layer: The `Layer` whose `call` is currently active.
    inputs: The inputs to the currently active `Layer`.
    build_graph: Whether currently inside a Graph or FuncGraph.
    training: Whether currently executing in training or inference mode.
    saving: Whether currently saving to SavedModel.
    frozen: Whether currently executing inside a `Layer` with `trainable` set to
      `False`.
    in_keras_graph: Whether executing inside the Keras Graph.
  """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.in_call = False
        self._state = {'layer': None, 'inputs': None, 'build_graph': False, 'training': None, 'saving': None}
        self._in_keras_graph = False

    def enter(self, layer, inputs, build_graph, training, saving=None):
        if False:
            print('Hello World!')
        'Push a Layer and its inputs and state onto the current call context.\n\n    Args:\n      layer: The `Layer` whose `call` is currently active.\n      inputs: The inputs to the currently active `Layer`.\n      build_graph: Whether currently inside a Graph or FuncGraph.\n      training: Whether currently executing in training or inference mode.\n      saving: Whether currently saving to SavedModel.\n\n    Returns:\n      Context manager.\n    '
        state = {'layer': layer, 'inputs': inputs, 'build_graph': build_graph, 'training': training, 'saving': saving}
        return CallContextManager(self, state)

    @property
    def layer(self):
        if False:
            while True:
                i = 10
        return self._state['layer']

    @property
    def inputs(self):
        if False:
            while True:
                i = 10
        return self._state['inputs']

    @property
    def build_graph(self):
        if False:
            for i in range(10):
                print('nop')
        return self._state['build_graph']

    @property
    def training(self):
        if False:
            print('Hello World!')
        return self._state['training']

    @property
    def saving(self):
        if False:
            i = 10
            return i + 15
        return self._state['saving']

    @property
    def frozen(self):
        if False:
            print('Hello World!')
        layer = self._state['layer']
        if not layer:
            return False
        return not layer.trainable

    @property
    def in_keras_graph(self):
        if False:
            print('Hello World!')
        if context.executing_eagerly():
            return False
        return self._in_keras_graph or getattr(backend.get_graph(), 'name', None) == 'keras_graph'

class CallContextManager(object):
    """Context manager for `CallContext`."""

    def __init__(self, call_ctx, state):
        if False:
            i = 10
            return i + 15
        self._call_ctx = call_ctx
        self._state = state
        self._build_graph = state['build_graph']

    def __enter__(self):
        if False:
            return 10
        call_ctx = self._call_ctx
        self._prev_in_call = call_ctx.in_call
        self._prev_state = call_ctx._state
        call_ctx.in_call = True
        call_ctx._state = self._state
        if self._build_graph:
            self._prev_in_keras_graph = call_ctx._in_keras_graph
            call_ctx._in_keras_graph = call_ctx._in_keras_graph or getattr(backend.get_graph(), 'name', None) == 'keras_graph'

    def __exit__(self, *exc_info):
        if False:
            return 10
        call_ctx = self._call_ctx
        call_ctx.in_call = self._prev_in_call
        call_ctx._state = self._prev_state
        if self._build_graph:
            call_ctx._in_keras_graph = self._prev_in_keras_graph

def training_arg_passed_to_call(argspec, args, kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Returns whether a user passed the `training` argument in `__call__`.'
    full_args = dict(zip(argspec.args[2:], args))
    full_args.update(kwargs)
    return 'training' in full_args and full_args['training'] is not None

def is_subclassed(layer):
    if False:
        i = 10
        return i + 15
    'Returns True if the object is a subclassed layer or subclassed model.'
    return layer.__module__.find('keras.engine') == -1 and layer.__module__.find('keras.layers') == -1

def from_saved_model(layer):
    if False:
        while True:
            i = 10
    'Returns whether the layer is loaded from a SavedModel.'
    return layer.__module__.find('keras.saving.saved_model') != -1

def check_graph_consistency(tensor=None, method='add_loss', force_raise=False):
    if False:
        return 10
    "Checks that tensors passed to `add_*` method match the Keras graph.\n\n  When one of the `add_*` method is called inside a V2 conditional branch,\n  the underlying tensor gets created in a FuncGraph managed by control_flow_v2.\n  We need to raise clear error messages in such cases.\n\n  Args:\n    tensor: Tensor to check, or `False` if it is known that an error\n      should be raised.\n    method: Caller method, one of {'add_metric', 'add_loss', 'add_update'}.\n    force_raise: If an error should be raised regardless of `tensor`.\n\n  Raises:\n    RuntimeError: In case of an out-of-graph tensor.\n  "
    if force_raise or (ops.executing_eagerly_outside_functions() and hasattr(tensor, 'graph') and tensor.graph.is_control_flow_graph):
        if method == 'activity_regularizer':
            bad_example = "\n      class TestModel(tf.keras.Model):\n\n        def __init__(self):\n          super(TestModel, self).__init__(name='test_model')\n          self.dense = tf.keras.layers.Dense(2, activity_regularizer='l2')\n\n        def call(self, x, training=None):\n          if training:\n            return self.dense(x)\n          else:\n            return self.dense(x)\n      "
            correct_example = "\n      class TestModel(tf.keras.Model):\n\n        def __init__(self):\n          super(TestModel, self).__init__(name='test_model')\n          self.dense = tf.keras.layers.Dense(2, activity_regularizer='l2')\n\n        def call(self, x, training=None):\n          return self.dense(x)\n      "
            raise RuntimeError('You are using a layer with `activity_regularizer` in a control flow branch, e.g.:\n{bad_example}\nThis is currently not supported. Please move your call to the layer with `activity_regularizer` out of the control flow branch, e.g.:\n{correct_example}\nYou can also resolve this by marking your outer model/layer dynamic (eager-only) by passing `dynamic=True` to the layer constructor. Any kind of control flow is supported with dynamic layers. Note that using `dynamic=True` requires you to implement static shape inference in the `compute_output_shape(input_shape)` method.'.format(bad_example=bad_example, correct_example=correct_example))
        if method == 'add_metric':
            bad_example = "\n      def call(self, inputs, training=None):\n        if training:\n          metric = compute_metric(inputs)\n          self.add_metric(metric, name='my_metric', aggregation='mean')\n        return inputs\n      "
            correct_example = "\n      def call(self, inputs, training=None):\n        if training:\n          metric = compute_metric(inputs)\n        else:\n          metric = 0.\n        self.add_metric(metric, name='my_metric', aggregation='mean')\n        return inputs\n      "
        elif method == 'add_loss':
            bad_example = '\n      def call(self, inputs, training=None):\n        if training:\n          loss = compute_loss(inputs)\n          self.add_loss(loss)\n        return inputs\n      '
            correct_example = '\n      def call(self, inputs, training=None):\n        if training:\n          loss = compute_loss(inputs)\n        else:\n          loss = 0.\n        self.add_loss(loss)\n        return inputs\n      '
        else:
            bad_example = '\n      def call(self, inputs, training=None):\n        if training:\n          self.add_update(self.w.assign_add(1))\n        return inputs\n      '
            correct_example = '\n      def call(self, inputs, training=None):\n        if training:\n          increment = 1\n        else:\n          increment = 0\n        self.add_update(self.w.assign_add(increment))\n        return inputs\n      '
        raise RuntimeError('You are using the method `{method}` in a control flow branch in your layer, e.g.:\n{bad_example}\nThis is not currently supported. Please move your call to {method} out of the control flow branch, e.g.:\n{correct_example}\nYou can also resolve this by marking your layer as dynamic (eager-only) by passing `dynamic=True` to the layer constructor. Any kind of control flow is supported with dynamic layers. Note that using `dynamic=True` requires you to implement static shape inference in the `compute_output_shape(input_shape)` method.'.format(method=method, bad_example=bad_example, correct_example=correct_example))

def mark_as_return(outputs, acd):
    if False:
        i = 10
        return i + 15
    'Marks `outputs` as the return values for automatic control deps.'

    def _mark_as_return(tensor):
        if False:
            print('Hello World!')
        'Marks `tensor` as the return value for automatic control deps.'
        if not tensor_util.is_tf_type(tensor):
            return tensor
        return_tensor = acd.mark_as_return(tensor)
        if getattr(tensor, '_keras_mask', None) is not None:
            return_tensor._keras_mask = acd.mark_as_return(tensor._keras_mask)
        else:
            return_tensor._keras_mask = None
        if getattr(tensor, '_tfp_distribution', None) is not None:
            return_tensor._tfp_distribution = tensor._tfp_distribution
        return return_tensor
    return nest.map_structure(_mark_as_return, outputs)
V2_DTYPE_BEHAVIOR = None

def enable_v2_dtype_behavior():
    if False:
        while True:
            i = 10
    "Enable the V2 dtype behavior for Keras layers.\n\n  By default, the V2 dtype behavior is enabled in TensorFlow 2, so this function\n  is only useful if `tf.compat.v1.disable_v2_behavior` has been called. Since\n  mixed precision requires V2 dtype behavior to be enabled, this function allows\n  you to use mixed precision in Keras layers if `disable_v2_behavior` has been\n  called.\n\n  When enabled, the dtype of Keras layers defaults to floatx (which is typically\n  float32) instead of None. In addition, layers will automatically cast\n  floating-point inputs to the layer's dtype.\n\n  >>> x = tf.ones((4, 4, 4, 4), dtype='float64')\n  >>> layer = tf.keras.layers.Conv2D(filters=4, kernel_size=2)\n  >>> print(layer.dtype)  # float32 since V2 dtype behavior is enabled\n  float32\n  >>> y = layer(x)  # Layer casts inputs since V2 dtype behavior is enabled\n  >>> print(y.dtype.name)\n  float32\n\n  A layer author can opt-out their layer from the automatic input casting by\n  passing `autocast=False` to the base Layer's constructor. This disables the\n  autocasting part of the V2 behavior for that layer, but not the defaulting to\n  floatx part of the V2 behavior.\n\n  When a global `tf.keras.mixed_precision.Policy` is set, a Keras layer's dtype\n  will default to the global policy instead of floatx. Layers will automatically\n  cast inputs to the policy's compute_dtype.\n  "
    global V2_DTYPE_BEHAVIOR
    V2_DTYPE_BEHAVIOR = True

def disable_v2_dtype_behavior():
    if False:
        return 10
    'Disables the V2 dtype behavior for Keras layers.\n\n  See `tf.compat.v1.keras.layers.enable_v2_dtype_behavior`.\n  '
    global V2_DTYPE_BEHAVIOR
    V2_DTYPE_BEHAVIOR = False

def v2_dtype_behavior_enabled():
    if False:
        return 10
    'Returns True if the V2 dtype behavior is enabled.'
    if V2_DTYPE_BEHAVIOR is None:
        return tf2.enabled()
    return V2_DTYPE_BEHAVIOR

class TrackableWeightHandler(object):
    """Keras wrapper for handling tracking.Trackable object saving and restoring.

  This class handles Trackables in both V1 and V2 modes, ensuring that they can
  be saved and restored with the correct data and without adding additional ops
  on every save.

  Attributes:
    trackable: The trackable to wrap.
    num_tensors: The number of tensors that this trackable requires for saving.
  """

    def __init__(self, trackable):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(trackable, tracking.Trackable):
            raise ValueError('%s is not a Trackable object.' % (trackable,))
        self._trackable = trackable
        self._distribute_strategy = distribute_lib.get_strategy()
        saveables = saveable_object_util.saveable_objects_from_trackable(trackable).values()
        if not saveables:
            self._num_tensors = 0
            self._setter = lambda weights: None
            self._getter = lambda : []
        elif len(saveables) == 1:
            saveable = list(saveables)[0]
            if ops.executing_eagerly_outside_functions():
                self._saveable = saveable
                self._num_tensors = len(self._saveable().specs)
                self._setter = lambda weights: self._saveable().restore(weights, None)
                self._getter = lambda : [spec.tensor for spec in self._saveable().specs]
            else:
                self._placeholder_tensors = []
                self._saveable = saveable()
                self._num_tensors = len(self._saveable.specs)
                for spec in self._saveable.specs:
                    tensor = spec.tensor
                    self._placeholder_tensors.append(array_ops.placeholder(tensor.dtype, tensor.shape))
                self._assign_op = self._saveable.restore(self._placeholder_tensors, None)
                self._setter = self._set_weights_v1
                self._getter = lambda : [spec.tensor for spec in self._saveable.specs]
        else:
            raise ValueError('Only Trackables with one Saveable are supported. The Trackable %s has %d Saveables.' % (trackable, len(saveables)))

    @property
    def num_tensors(self):
        if False:
            for i in range(10):
                print('nop')
        return self._num_tensors

    def set_weights(self, weights):
        if False:
            return 10
        if len(weights) != self._num_tensors:
            raise ValueError(('Weight handler for trackable %s received the wrong number of ' + 'weights: expected %s, got %s.') % (self._trackable, self._num_tensors, len(weights)))
        self._setter(weights)

    def get_tensors(self):
        if False:
            i = 10
            return i + 15
        return self._getter()

    def _set_weights_v1(self, weights):
        if False:
            for i in range(10):
                print('nop')
        feed_dict = {}
        for (idx, tensor) in enumerate(weights):
            feed_dict[self._placeholder_tensors[idx]] = tensor
        backend.get_session().run(self._assign_op, feed_dict)

class StaticTableHandler(TrackableWeightHandler):
    """Wrapper for handling weight collection for static hash tables."""

    def __init__(self, getter_lambda):
        if False:
            for i in range(10):
                print('nop')
        self._num_tensors = 2
        self._getter = getter_lambda
        self._distribute_strategy = distribute_lib.get_strategy()

        def raise_error(_):
            if False:
                return 10
            raise RuntimeError('This layer contains a static lookup table, which cannot be changed via set_weights().')
        self._setter = raise_error

def no_ragged_support(inputs, layer_name):
    if False:
        return 10
    input_list = nest.flatten(inputs)
    if any((isinstance(x, ragged_tensor.RaggedTensor) for x in input_list)):
        raise ValueError('Layer %s does not support RaggedTensors as input. Inputs received: %s. You can try converting your input to an uniform tensor.' % (layer_name, inputs))

def is_split_variable(v):
    if False:
        for i in range(10):
            print('nop')
    'Returns True if `v` is either a PartionedVariable or a ShardedVariable.'
    return hasattr(v, '_variable_list') or hasattr(v, '_variables')

def has_weights(obj):
    if False:
        return 10
    obj_type = type(obj)
    return hasattr(obj_type, 'trainable_weights') and hasattr(obj_type, 'non_trainable_weights') and (not isinstance(obj, type))
REVIVED_LOSS_PLACEHOLDER = "This layer's losses have been added to the parent layer."