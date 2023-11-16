"""RefVariable class."""
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import resource_variables_toggle
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated

def default_variable_creator(next_creator=None, **kwargs):
    if False:
        i = 10
        return i + 15
    'Default variable creator.'
    assert next_creator is None
    initial_value = kwargs.get('initial_value', None)
    trainable = kwargs.get('trainable', None)
    collections = kwargs.get('collections', None)
    validate_shape = kwargs.get('validate_shape', True)
    caching_device = kwargs.get('caching_device', None)
    name = kwargs.get('name', None)
    variable_def = kwargs.get('variable_def', None)
    dtype = kwargs.get('dtype', None)
    expected_shape = kwargs.get('expected_shape', None)
    import_scope = kwargs.get('import_scope', None)
    constraint = kwargs.get('constraint', None)
    use_resource = kwargs.get('use_resource', None)
    synchronization = kwargs.get('synchronization', None)
    aggregation = kwargs.get('aggregation', None)
    shape = kwargs.get('shape', None)
    if use_resource is None:
        use_resource = variable_scope.get_variable_scope().use_resource
    if use_resource is None:
        use_resource = resource_variables_toggle.resource_variables_enabled()
    use_resource = use_resource or context.executing_eagerly()
    if use_resource:
        distribute_strategy = kwargs.get('distribute_strategy', None)
        return resource_variable_ops.ResourceVariable(initial_value=initial_value, trainable=trainable, collections=collections, validate_shape=validate_shape, caching_device=caching_device, name=name, dtype=dtype, constraint=constraint, variable_def=variable_def, import_scope=import_scope, distribute_strategy=distribute_strategy, synchronization=synchronization, aggregation=aggregation, shape=shape)
    else:
        return RefVariable(initial_value=initial_value, trainable=trainable, collections=collections, validate_shape=validate_shape, caching_device=caching_device, name=name, dtype=dtype, constraint=constraint, variable_def=variable_def, expected_shape=expected_shape, import_scope=import_scope, synchronization=synchronization, aggregation=aggregation, shape=shape)
variable_v1.default_variable_creator = default_variable_creator

def _to_proto_fn(v, export_scope=None):
    if False:
        while True:
            i = 10
    'Converts Variable and ResourceVariable to VariableDef for collections.'
    return v.to_proto(export_scope=export_scope)

def _from_proto_fn(v, import_scope=None):
    if False:
        for i in range(10):
            print('nop')
    'Creates Variable or ResourceVariable from VariableDef as needed.'
    if v.is_resource:
        return resource_variable_ops.ResourceVariable.from_proto(v, import_scope=import_scope)
    return variable_v1.VariableV1.from_proto(v, import_scope=import_scope)
ops.register_proto_function(ops.GraphKeys.GLOBAL_VARIABLES, proto_type=variable_pb2.VariableDef, to_proto=_to_proto_fn, from_proto=_from_proto_fn)
ops.register_proto_function(ops.GraphKeys.TRAINABLE_VARIABLES, proto_type=variable_pb2.VariableDef, to_proto=_to_proto_fn, from_proto=_from_proto_fn)
ops.register_proto_function(ops.GraphKeys.MOVING_AVERAGE_VARIABLES, proto_type=variable_pb2.VariableDef, to_proto=_to_proto_fn, from_proto=_from_proto_fn)
ops.register_proto_function(ops.GraphKeys.LOCAL_VARIABLES, proto_type=variable_pb2.VariableDef, to_proto=_to_proto_fn, from_proto=_from_proto_fn)
ops.register_proto_function(ops.GraphKeys.MODEL_VARIABLES, proto_type=variable_pb2.VariableDef, to_proto=_to_proto_fn, from_proto=_from_proto_fn)
ops.register_proto_function(ops.GraphKeys.GLOBAL_STEP, proto_type=variable_pb2.VariableDef, to_proto=_to_proto_fn, from_proto=_from_proto_fn)
ops.register_proto_function(ops.GraphKeys.METRIC_VARIABLES, proto_type=variable_pb2.VariableDef, to_proto=_to_proto_fn, from_proto=_from_proto_fn)

class RefVariable(variable_v1.VariableV1, core.Tensor):
    """Ref-based implementation of variables."""

    def __init__(self, initial_value=None, trainable=None, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None, expected_shape=None, import_scope=None, constraint=None, synchronization=None, aggregation=None, shape=None):
        if False:
            i = 10
            return i + 15
        "Creates a new variable with value `initial_value`.\n\n    The new variable is added to the graph collections listed in `collections`,\n    which defaults to `[GraphKeys.GLOBAL_VARIABLES]`.\n\n    If `trainable` is `True` the variable is also added to the graph collection\n    `GraphKeys.TRAINABLE_VARIABLES`.\n\n    This constructor creates both a `variable` Op and an `assign` Op to set the\n    variable to its initial value.\n\n    Args:\n      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,\n        which is the initial value for the Variable. The initial value must have\n        a shape specified unless `validate_shape` is set to False. Can also be a\n        callable with no argument that returns the initial value when called. In\n        that case, `dtype` must be specified. (Note that initializer functions\n        from init_ops.py must first be bound to a shape before being used here.)\n      trainable: If `True`, also adds the variable to the graph collection\n        `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as the default\n        list of variables to use by the `Optimizer` classes. Defaults to `True`,\n        unless `synchronization` is set to `ON_READ`, in which case it defaults\n        to `False`.\n      collections: List of graph collections keys. The new variable is added to\n        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.\n      validate_shape: If `False`, allows the variable to be initialized with a\n        value of unknown shape. If `True`, the default, the shape of\n        `initial_value` must be known.\n      caching_device: Optional device string describing where the Variable\n        should be cached for reading.  Defaults to the Variable's device. If not\n        `None`, caches on another device.  Typical use is to cache on the device\n        where the Ops using the Variable reside, to deduplicate copying through\n        `Switch` and other conditional statements.\n      name: Optional name for the variable. Defaults to `'Variable'` and gets\n        uniquified automatically.\n      variable_def: `VariableDef` protocol buffer. If not `None`, recreates the\n        Variable object with its contents, referencing the variable's nodes in\n        the graph, which must already exist. The graph is not changed.\n        `variable_def` and the other arguments are mutually exclusive.\n      dtype: If set, initial_value will be converted to the given type. If\n        `None`, either the datatype will be kept (if `initial_value` is a\n        Tensor), or `convert_to_tensor` will decide.\n      expected_shape: A TensorShape. If set, initial_value is expected to have\n        this shape.\n      import_scope: Optional `string`. Name scope to add to the `Variable.` Only\n        used when initializing from protocol buffer.\n      constraint: An optional projection function to be applied to the variable\n        after being updated by an `Optimizer` (e.g. used to implement norm\n        constraints or value constraints for layer weights). The function must\n        take as input the unprojected Tensor representing the value of the\n        variable and return the Tensor for the projected value (which must have\n        the same shape). Constraints are not safe to use when doing asynchronous\n        distributed training.\n      synchronization: Indicates when a distributed a variable will be\n        aggregated. Accepted values are constants defined in the class\n        `tf.VariableSynchronization`. By default the synchronization is set to\n        `AUTO` and the current `DistributionStrategy` chooses when to\n        synchronize.\n      aggregation: Indicates how a distributed variable will be aggregated.\n        Accepted values are constants defined in the class\n        `tf.VariableAggregation`.\n      shape: (optional) The shape of this variable. If None, the shape of\n        `initial_value` will be used. When setting this argument to\n        `tf.TensorShape(None)` (representing an unspecified shape), the variable\n        can be assigned with values of different shapes.\n\n    Raises:\n      ValueError: If both `variable_def` and initial_value are specified.\n      ValueError: If the initial value is not specified, or does not have a\n        shape and `validate_shape` is `True`.\n      RuntimeError: If eager execution is enabled.\n    "
        self._in_graph_mode = True
        if variable_def:
            if initial_value:
                raise ValueError('variable_def and initial_value are mutually exclusive.')
            self._init_from_proto(variable_def, import_scope=import_scope)
        else:
            self._init_from_args(initial_value=initial_value, trainable=trainable, collections=collections, validate_shape=validate_shape, caching_device=caching_device, name=name, dtype=dtype, expected_shape=expected_shape, constraint=constraint, synchronization=synchronization, aggregation=aggregation, shape=shape)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        if context.executing_eagerly() and (not self._in_graph_mode):
            return "<tf.Variable '%s' shape=%s dtype=%s, numpy=%s>" % (self.name, self.get_shape(), self.dtype.name, ops.numpy_text(self.read_value(), is_repr=True))
        else:
            return "<tf.Variable '%s' shape=%s dtype=%s>" % (self.name, self.get_shape(), self.dtype.name)

    def _init_from_args(self, initial_value=None, trainable=None, collections=None, validate_shape=True, caching_device=None, name=None, dtype=None, expected_shape=None, constraint=None, synchronization=None, aggregation=None, shape=None):
        if False:
            print('Hello World!')
        "Creates a new variable from arguments.\n\n    Args:\n      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,\n        which is the initial value for the Variable. The initial value must have\n        a shape specified unless `validate_shape` is set to False. Can also be a\n        callable with no argument that returns the initial value when called.\n        (Note that initializer functions from init_ops.py must first be bound to\n        a shape before being used here.)\n      trainable: If `True`, also adds the variable to the graph collection\n        `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as the default\n        list of variables to use by the `Optimizer` classes. Defaults to `True`,\n        unless `synchronization` is set to `ON_READ`, in which case it defaults\n        to `False`.\n      collections: List of graph collections keys. The new variable is added to\n        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.\n      validate_shape: If `False`, allows the variable to be initialized with a\n        value of unknown shape. If `True`, the default, the shape of\n        `initial_value` must be known.\n      caching_device: Optional device string or function describing where the\n        Variable should be cached for reading.  Defaults to the Variable's\n        device.  If not `None`, caches on another device.  Typical use is to\n        cache on the device where the Ops using the Variable reside, to\n        deduplicate copying through `Switch` and other conditional statements.\n      name: Optional name for the variable. Defaults to `'Variable'` and gets\n        uniquified automatically.\n      dtype: If set, initial_value will be converted to the given type. If None,\n        either the datatype will be kept (if initial_value is a Tensor) or\n        float32 will be used (if it is a Python object convertible to a Tensor).\n      expected_shape: Deprecated. Ignored.\n      constraint: An optional projection function to be applied to the variable\n        after being updated by an `Optimizer` (e.g. used to implement norm\n        constraints or value constraints for layer weights). The function must\n        take as input the unprojected Tensor representing the value of the\n        variable and return the Tensor for the projected value (which must have\n        the same shape). Constraints are not safe to use when doing asynchronous\n        distributed training.\n      synchronization: Indicates when a distributed a variable will be\n        aggregated. Accepted values are constants defined in the class\n        `tf.VariableSynchronization`. By default the synchronization is set to\n        `AUTO` and the current `DistributionStrategy` chooses when to\n        synchronize.\n      aggregation: Indicates how a distributed variable will be aggregated.\n        Accepted values are constants defined in the class\n        `tf.VariableAggregation`.\n      shape: (optional) The shape of this variable. If None, the shape of\n        `initial_value` will be used. When setting this argument to\n        `tf.TensorShape(None)` (representing an unspecified shape), the variable\n        can be assigned with values of different shapes.\n\n    Raises:\n      ValueError: If the initial value is not specified, or does not have a\n        shape and `validate_shape` is `True`.\n      RuntimeError: If lifted into the eager context.\n    "
        _ = expected_shape
        if initial_value is None:
            raise ValueError('initial_value must be specified.')
        init_from_fn = callable(initial_value)
        if collections is None:
            collections = [ops.GraphKeys.GLOBAL_VARIABLES]
        if not isinstance(collections, (list, tuple, set)):
            raise ValueError('collections argument to Variable constructor must be a list, tuple, or set. Got %s of type %s' % (collections, type(collections)))
        if constraint is not None and (not callable(constraint)):
            raise ValueError('The `constraint` argument must be a callable.')
        self._graph_key = ops.get_default_graph()._graph_key
        if isinstance(initial_value, trackable.CheckpointInitialValue):
            self._maybe_initialize_trackable()
            self._update_uid = initial_value.checkpoint_position.restore_uid
            initial_value = initial_value.wrapped_value
        (synchronization, aggregation, trainable) = variables.validate_synchronization_aggregation_trainable(synchronization, aggregation, trainable, name)
        self._synchronization = synchronization
        self._aggregation = aggregation
        self._trainable = trainable
        if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
            collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]
        with ops.init_scope():
            if context.executing_eagerly():
                raise RuntimeError('Reference variables are not supported when eager execution is enabled. Please run `tf.compat.v1.enable_resource_variables()` to switch to resource variables.')
            with ops.name_scope(name, 'Variable', [] if init_from_fn else [initial_value]) as name:
                if init_from_fn:
                    true_name = ops.name_from_scope_name(name)
                    attr = attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(s=[compat.as_bytes('loc:@%s' % true_name)]))
                    with ops.get_default_graph()._attr_scope({'_class': attr}):
                        with ops.name_scope('Initializer'), ops.device(None):
                            initial_value = initial_value()
                            if isinstance(initial_value, trackable.CheckpointInitialValue):
                                self._maybe_initialize_trackable()
                                self._update_uid = initial_value.checkpoint_position.restore_uid
                                initial_value = initial_value.wrapped_value
                            self._initial_value = ops.convert_to_tensor(initial_value, name='initial_value', dtype=dtype)
                            if shape is None:
                                shape = self._initial_value.get_shape() if validate_shape else tensor_shape.unknown_shape()
                        self._variable = state_ops.variable_op_v2(shape, self._initial_value.dtype.base_dtype, name=name)
                else:
                    self._initial_value = ops.convert_to_tensor(initial_value, name='initial_value', dtype=dtype)
                    if self._initial_value.op._get_control_flow_context() is not None:
                        raise ValueError('Initializer for variable %s is from inside a control-flow construct, such as a loop or conditional. When creating a variable inside a loop or conditional, use a lambda as the initializer.' % name)
                    if shape is None:
                        shape = self._initial_value.get_shape() if validate_shape else tensor_shape.unknown_shape()
                    self._variable = state_ops.variable_op_v2(shape, self._initial_value.dtype.base_dtype, name=name)
                self._name = self._variable.name
                if validate_shape:
                    initial_value_shape = self._initial_value.get_shape()
                    if not initial_value_shape.is_fully_defined():
                        raise ValueError('initial_value must have a shape specified: %s' % self._initial_value)
                self._initializer_op = state_ops.assign(self._variable, variables._try_guard_against_uninitialized_dependencies(name, self._initial_value), validate_shape=validate_shape).op
                if caching_device is not None:
                    with ops.device(caching_device):
                        self._snapshot = array_ops.identity(self._variable, name='read')
                else:
                    with ops.colocate_with(self._variable.op):
                        self._snapshot = array_ops.identity(self._variable, name='read')
            ops.add_to_collections(collections, self)
        self._caching_device = caching_device
        self._save_slice_info = None
        self._constraint = constraint

    def _init_from_proto(self, variable_def, import_scope=None):
        if False:
            while True:
                i = 10
        'Recreates the Variable object from a `VariableDef` protocol buffer.\n\n    Args:\n      variable_def: `VariableDef` protocol buffer, describing a variable whose\n        nodes already exists in the graph.\n      import_scope: Optional `string`. Name scope to add.\n    '
        assert isinstance(variable_def, variable_pb2.VariableDef)
        g = ops.get_default_graph()
        self._variable = g.as_graph_element(ops.prepend_name_scope(variable_def.variable_name, import_scope=import_scope))
        self._name = self._variable.name
        self._initializer_op = g.as_graph_element(ops.prepend_name_scope(variable_def.initializer_name, import_scope=import_scope))
        if hasattr(variable_def, 'initial_value_name') and variable_def.initial_value_name:
            self._initial_value = g.as_graph_element(ops.prepend_name_scope(variable_def.initial_value_name, import_scope=import_scope))
        else:
            self._initial_value = None
        (synchronization, aggregation, trainable) = variables.validate_synchronization_aggregation_trainable(variable_def.synchronization, variable_def.aggregation, variable_def.trainable, variable_def.variable_name)
        self._synchronization = synchronization
        self._aggregation = aggregation
        self._trainable = trainable
        self._snapshot = g.as_graph_element(ops.prepend_name_scope(variable_def.snapshot_name, import_scope=import_scope))
        if variable_def.HasField('save_slice_info_def'):
            self._save_slice_info = variables.Variable.SaveSliceInfo(save_slice_info_def=variable_def.save_slice_info_def, import_scope=import_scope)
        else:
            self._save_slice_info = None
        self._caching_device = None
        self._constraint = None

    def _as_graph_element(self):
        if False:
            while True:
                i = 10
        'Conversion function for Graph.as_graph_element().'
        return self._variable

    def value(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the last snapshot of this variable.\n\n    You usually do not need to call this method as all ops that need the value\n    of the variable call it automatically through a `convert_to_tensor()` call.\n\n    Returns a `Tensor` which holds the value of the variable.  You can not\n    assign a new value to this tensor as it is not a reference to the variable.\n\n    To avoid copies, if the consumer of the returned value is on the same device\n    as the variable, this actually returns the live value of the variable, not\n    a copy.  Updates to the variable are seen by the consumer.  If the consumer\n    is on a different device it will get a copy of the variable.\n\n    Returns:\n      A `Tensor` containing the value of the variable.\n    '
        return self._snapshot

    def read_value(self):
        if False:
            while True:
                i = 10
        "Returns the value of this variable, read in the current context.\n\n    Can be different from value() if it's on another device, with control\n    dependencies, etc.\n\n    Returns:\n      A `Tensor` containing the value of the variable.\n    "
        return array_ops.identity(self._variable, name='read')

    def _ref(self):
        if False:
            i = 10
            return i + 15
        'Returns a reference to this variable.\n\n    You usually do not need to call this method as all ops that need a reference\n    to the variable call it automatically.\n\n    Returns is a `Tensor` which holds a reference to the variable.  You can\n    assign a new value to the variable by passing the tensor to an assign op.\n    See `tf.Variable.value` if you want to get the value of the\n    variable.\n\n    Returns:\n      A `Tensor` that is a reference to the variable.\n    '
        return self._variable

    def set_shape(self, shape):
        if False:
            while True:
                i = 10
        'Overrides the shape for this variable.\n\n    Args:\n      shape: the `TensorShape` representing the overridden shape.\n    '
        self._ref().set_shape(shape)
        self.value().set_shape(shape)

    @property
    def trainable(self):
        if False:
            i = 10
            return i + 15
        return self._trainable

    @property
    def synchronization(self):
        if False:
            while True:
                i = 10
        return self._synchronization

    @property
    def aggregation(self):
        if False:
            for i in range(10):
                print('nop')
        return self._aggregation

    def eval(self, session=None):
        if False:
            i = 10
            return i + 15
        "In a session, computes and returns the value of this variable.\n\n    This is not a graph construction method, it does not add ops to the graph.\n\n    This convenience method requires a session where the graph\n    containing this variable has been launched. If no session is\n    passed, the default session is used.  See `tf.compat.v1.Session` for more\n    information on launching a graph and on sessions.\n\n    ```python\n    v = tf.Variable([1, 2])\n    init = tf.compat.v1.global_variables_initializer()\n\n    with tf.compat.v1.Session() as sess:\n        sess.run(init)\n        # Usage passing the session explicitly.\n        print(v.eval(sess))\n        # Usage with the default session.  The 'with' block\n        # above makes 'sess' the default session.\n        print(v.eval())\n    ```\n\n    Args:\n      session: The session to use to evaluate this variable. If none, the\n        default session is used.\n\n    Returns:\n      A numpy `ndarray` with a copy of the value of this variable.\n    "
        return self._variable.eval(session=session)

    @property
    def initial_value(self):
        if False:
            print('Hello World!')
        'Returns the Tensor used as the initial value for the variable.\n\n    Note that this is different from `initialized_value()` which runs\n    the op that initializes the variable before returning its value.\n    This method returns the tensor that is used by the op that initializes\n    the variable.\n\n    Returns:\n      A `Tensor`.\n    '
        return self._initial_value

    @property
    def constraint(self):
        if False:
            while True:
                i = 10
        'Returns the constraint function associated with this variable.\n\n    Returns:\n      The constraint function that was passed to the variable constructor.\n      Can be `None` if no constraint was passed.\n    '
        return self._constraint

    def assign(self, value, use_locking=False, name=None, read_value=True):
        if False:
            while True:
                i = 10
        'Assigns a new value to the variable.\n\n    This is essentially a shortcut for `assign(self, value)`.\n\n    Args:\n      value: A `Tensor`. The new value for this variable.\n      use_locking: If `True`, use locking during the assignment.\n      name: The name of the operation to be created\n      read_value: if True, will return something which evaluates to the new\n        value of the variable; if False will return the assign op.\n\n    Returns:\n      A `Tensor` that will hold the new value of this variable after\n      the assignment has completed.\n    '
        assign = state_ops.assign(self._variable, value, use_locking=use_locking, name=name)
        if read_value:
            return assign
        return assign.op

    def assign_add(self, delta, use_locking=False, name=None, read_value=True):
        if False:
            while True:
                i = 10
        'Adds a value to this variable.\n\n     This is essentially a shortcut for `assign_add(self, delta)`.\n\n    Args:\n      delta: A `Tensor`. The value to add to this variable.\n      use_locking: If `True`, use locking during the operation.\n      name: The name of the operation to be created\n      read_value: if True, will return something which evaluates to the new\n        value of the variable; if False will return the assign op.\n\n    Returns:\n      A `Tensor` that will hold the new value of this variable after\n      the addition has completed.\n    '
        assign = state_ops.assign_add(self._variable, delta, use_locking=use_locking, name=name)
        if read_value:
            return assign
        return assign.op

    def assign_sub(self, delta, use_locking=False, name=None, read_value=True):
        if False:
            for i in range(10):
                print('nop')
        'Subtracts a value from this variable.\n\n    This is essentially a shortcut for `assign_sub(self, delta)`.\n\n    Args:\n      delta: A `Tensor`. The value to subtract from this variable.\n      use_locking: If `True`, use locking during the operation.\n      name: The name of the operation to be created\n      read_value: if True, will return something which evaluates to the new\n        value of the variable; if False will return the assign op.\n\n    Returns:\n      A `Tensor` that will hold the new value of this variable after\n      the subtraction has completed.\n    '
        assign = state_ops.assign_sub(self._variable, delta, use_locking=use_locking, name=name)
        if read_value:
            return assign
        return assign.op

    def scatter_sub(self, sparse_delta, use_locking=False, name=None):
        if False:
            i = 10
            return i + 15
        'Subtracts `tf.IndexedSlices` from this variable.\n\n    Args:\n      sparse_delta: `tf.IndexedSlices` to be subtracted from this variable.\n      use_locking: If `True`, use locking during the operation.\n      name: the name of the operation.\n\n    Returns:\n      A `Tensor` that will hold the new value of this variable after\n      the scattered subtraction has completed.\n\n    Raises:\n      TypeError: if `sparse_delta` is not an `IndexedSlices`.\n    '
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError('sparse_delta is not IndexedSlices: %s' % sparse_delta)
        return gen_state_ops.scatter_sub(self._variable, sparse_delta.indices, sparse_delta.values, use_locking=use_locking, name=name)

    def scatter_add(self, sparse_delta, use_locking=False, name=None):
        if False:
            return 10
        'Adds `tf.IndexedSlices` to this variable.\n\n    Args:\n      sparse_delta: `tf.IndexedSlices` to be added to this variable.\n      use_locking: If `True`, use locking during the operation.\n      name: the name of the operation.\n\n    Returns:\n      A `Tensor` that will hold the new value of this variable after\n      the scattered addition has completed.\n\n    Raises:\n      TypeError: if `sparse_delta` is not an `IndexedSlices`.\n    '
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError('sparse_delta is not IndexedSlices: %s' % sparse_delta)
        return gen_state_ops.scatter_add(self._variable, sparse_delta.indices, sparse_delta.values, use_locking=use_locking, name=name)

    def scatter_max(self, sparse_delta, use_locking=False, name=None):
        if False:
            return 10
        'Updates this variable with the max of `tf.IndexedSlices` and itself.\n\n    Args:\n      sparse_delta: `tf.IndexedSlices` to use as an argument of max with this\n        variable.\n      use_locking: If `True`, use locking during the operation.\n      name: the name of the operation.\n\n    Returns:\n      A `Tensor` that will hold the new value of this variable after\n      the scattered maximization has completed.\n\n    Raises:\n      TypeError: if `sparse_delta` is not an `IndexedSlices`.\n    '
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError('sparse_delta is not IndexedSlices: %s' % sparse_delta)
        return gen_state_ops.scatter_max(self._variable, sparse_delta.indices, sparse_delta.values, use_locking=use_locking, name=name)

    def scatter_min(self, sparse_delta, use_locking=False, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Updates this variable with the min of `tf.IndexedSlices` and itself.\n\n    Args:\n      sparse_delta: `tf.IndexedSlices` to use as an argument of min with this\n        variable.\n      use_locking: If `True`, use locking during the operation.\n      name: the name of the operation.\n\n    Returns:\n      A `Tensor` that will hold the new value of this variable after\n      the scattered minimization has completed.\n\n    Raises:\n      TypeError: if `sparse_delta` is not an `IndexedSlices`.\n    '
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError('sparse_delta is not IndexedSlices: %s' % sparse_delta)
        return gen_state_ops.scatter_min(self._variable, sparse_delta.indices, sparse_delta.values, use_locking=use_locking, name=name)

    def scatter_mul(self, sparse_delta, use_locking=False, name=None):
        if False:
            i = 10
            return i + 15
        'Multiply this variable by `tf.IndexedSlices`.\n\n    Args:\n      sparse_delta: `tf.IndexedSlices` to multiply this variable by.\n      use_locking: If `True`, use locking during the operation.\n      name: the name of the operation.\n\n    Returns:\n      A `Tensor` that will hold the new value of this variable after\n      the scattered multiplication has completed.\n\n    Raises:\n      TypeError: if `sparse_delta` is not an `IndexedSlices`.\n    '
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError('sparse_delta is not IndexedSlices: %s' % sparse_delta)
        return gen_state_ops.scatter_mul(self._variable, sparse_delta.indices, sparse_delta.values, use_locking=use_locking, name=name)

    def scatter_div(self, sparse_delta, use_locking=False, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Divide this variable by `tf.IndexedSlices`.\n\n    Args:\n      sparse_delta: `tf.IndexedSlices` to divide this variable by.\n      use_locking: If `True`, use locking during the operation.\n      name: the name of the operation.\n\n    Returns:\n      A `Tensor` that will hold the new value of this variable after\n      the scattered division has completed.\n\n    Raises:\n      TypeError: if `sparse_delta` is not an `IndexedSlices`.\n    '
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError('sparse_delta is not IndexedSlices: %s' % sparse_delta)
        return gen_state_ops.scatter_div(self._variable, sparse_delta.indices, sparse_delta.values, use_locking=use_locking, name=name)

    def scatter_update(self, sparse_delta, use_locking=False, name=None):
        if False:
            return 10
        'Assigns `tf.IndexedSlices` to this variable.\n\n    Args:\n      sparse_delta: `tf.IndexedSlices` to be assigned to this variable.\n      use_locking: If `True`, use locking during the operation.\n      name: the name of the operation.\n\n    Returns:\n      A `Tensor` that will hold the new value of this variable after\n      the scattered assignment has completed.\n\n    Raises:\n      TypeError: if `sparse_delta` is not an `IndexedSlices`.\n    '
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError('sparse_delta is not IndexedSlices: %s' % sparse_delta)
        return gen_state_ops.scatter_update(self._variable, sparse_delta.indices, sparse_delta.values, use_locking=use_locking, name=name)

    def batch_scatter_update(self, sparse_delta, use_locking=False, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Assigns `tf.IndexedSlices` to this variable batch-wise.\n\n    Analogous to `batch_gather`. This assumes that this variable and the\n    sparse_delta IndexedSlices have a series of leading dimensions that are the\n    same for all of them, and the updates are performed on the last dimension of\n    indices. In other words, the dimensions should be the following:\n\n    `num_prefix_dims = sparse_delta.indices.ndims - 1`\n    `batch_dim = num_prefix_dims + 1`\n    `sparse_delta.updates.shape = sparse_delta.indices.shape + var.shape[\n         batch_dim:]`\n\n    where\n\n    `sparse_delta.updates.shape[:num_prefix_dims]`\n    `== sparse_delta.indices.shape[:num_prefix_dims]`\n    `== var.shape[:num_prefix_dims]`\n\n    And the operation performed can be expressed as:\n\n    `var[i_1, ..., i_n,\n         sparse_delta.indices[i_1, ..., i_n, j]] = sparse_delta.updates[\n            i_1, ..., i_n, j]`\n\n    When sparse_delta.indices is a 1D tensor, this operation is equivalent to\n    `scatter_update`.\n\n    To avoid this operation one can looping over the first `ndims` of the\n    variable and using `scatter_update` on the subtensors that result of slicing\n    the first dimension. This is a valid option for `ndims = 1`, but less\n    efficient than this implementation.\n\n    Args:\n      sparse_delta: `tf.IndexedSlices` to be assigned to this variable.\n      use_locking: If `True`, use locking during the operation.\n      name: the name of the operation.\n\n    Returns:\n      A `Tensor` that will hold the new value of this variable after\n      the scattered assignment has completed.\n\n    Raises:\n      TypeError: if `sparse_delta` is not an `IndexedSlices`.\n    '
        return state_ops.batch_scatter_update(self, sparse_delta.indices, sparse_delta.values, use_locking=use_locking, name=name)

    def scatter_nd_sub(self, indices, updates, name=None):
        if False:
            i = 10
            return i + 15
        'Applies sparse subtraction to individual values or slices in a Variable.\n\n    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.\n\n    `indices` must be integer tensor, containing indices into `ref`.\n    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.\n\n    The innermost dimension of `indices` (with length `K`) corresponds to\n    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th\n    dimension of `ref`.\n\n    `updates` is `Tensor` of rank `Q-1+P-K` with shape:\n\n    ```\n    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].\n    ```\n\n    For example, say we want to add 4 scattered elements to a rank-1 tensor to\n    8 elements. In Python, that update would look like this:\n\n    ```python\n        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])\n        indices = tf.constant([[4], [3], [1] ,[7]])\n        updates = tf.constant([9, 10, 11, 12])\n        op = ref.scatter_nd_sub(indices, updates)\n        with tf.compat.v1.Session() as sess:\n          print sess.run(op)\n    ```\n\n    The resulting update to ref would look like this:\n\n        [1, -9, 3, -6, -6, 6, 7, -4]\n\n    See `tf.scatter_nd` for more details about how to make updates to\n    slices.\n\n    Args:\n      indices: The indices to be used in the operation.\n      updates: The values to be used in the operation.\n      name: the name of the operation.\n\n    Returns:\n      A `Tensor` that will hold the new value of this variable after\n      the scattered subtraction has completed.\n    '
        return gen_state_ops.scatter_nd_sub(self._variable, indices, updates, use_locking=True, name=name)

    def scatter_nd_add(self, indices, updates, name=None):
        if False:
            i = 10
            return i + 15
        'Applies sparse addition to individual values or slices in a Variable.\n\n    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.\n\n    `indices` must be integer tensor, containing indices into `ref`.\n    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.\n\n    The innermost dimension of `indices` (with length `K`) corresponds to\n    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th\n    dimension of `ref`.\n\n    `updates` is `Tensor` of rank `Q-1+P-K` with shape:\n\n    ```\n    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].\n    ```\n\n    For example, say we want to add 4 scattered elements to a rank-1 tensor to\n    8 elements. In Python, that update would look like this:\n\n    ```python\n        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])\n        indices = tf.constant([[4], [3], [1] ,[7]])\n        updates = tf.constant([9, 10, 11, 12])\n        add = ref.scatter_nd_add(indices, updates)\n        with tf.compat.v1.Session() as sess:\n          print sess.run(add)\n    ```\n\n    The resulting update to ref would look like this:\n\n        [1, 13, 3, 14, 14, 6, 7, 20]\n\n    See `tf.scatter_nd` for more details about how to make updates to\n    slices.\n\n    Args:\n      indices: The indices to be used in the operation.\n      updates: The values to be used in the operation.\n      name: the name of the operation.\n\n    Returns:\n      A `Tensor` that will hold the new value of this variable after\n      the scattered addition has completed.\n    '
        return gen_state_ops.scatter_nd_add(self._variable, indices, updates, use_locking=True, name=name)

    def scatter_nd_update(self, indices, updates, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Applies sparse assignment to individual values or slices in a Variable.\n\n    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.\n\n    `indices` must be integer tensor, containing indices into `ref`.\n    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.\n\n    The innermost dimension of `indices` (with length `K`) corresponds to\n    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th\n    dimension of `ref`.\n\n    `updates` is `Tensor` of rank `Q-1+P-K` with shape:\n\n    ```\n    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].\n    ```\n\n    For example, say we want to add 4 scattered elements to a rank-1 tensor to\n    8 elements. In Python, that update would look like this:\n\n    ```python\n        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])\n        indices = tf.constant([[4], [3], [1] ,[7]])\n        updates = tf.constant([9, 10, 11, 12])\n        op = ref.scatter_nd_update(indices, updates)\n        with tf.compat.v1.Session() as sess:\n          print sess.run(op)\n    ```\n\n    The resulting update to ref would look like this:\n\n        [1, 11, 3, 10, 9, 6, 7, 12]\n\n    See `tf.scatter_nd` for more details about how to make updates to\n    slices.\n\n    Args:\n      indices: The indices to be used in the operation.\n      updates: The values to be used in the operation.\n      name: the name of the operation.\n\n    Returns:\n      A `Tensor` that will hold the new value of this variable after\n      the scattered assignment has completed.\n    '
        return gen_state_ops.scatter_nd_update(self._variable, indices, updates, use_locking=True, name=name)

    def scatter_nd_max(self, indices, updates, name=None):
        if False:
            while True:
                i = 10
        'Updates this variable with the max of `tf.IndexedSlices` and itself.\n\n    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.\n\n    `indices` must be integer tensor, containing indices into `ref`.\n    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.\n\n    The innermost dimension of `indices` (with length `K`) corresponds to\n    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th\n    dimension of `ref`.\n\n    `updates` is `Tensor` of rank `Q-1+P-K` with shape:\n\n    ```\n    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].\n    ```\n\n    See `tf.scatter_nd` for more details about how to make updates to\n    slices.\n\n    Args:\n      indices: The indices to be used in the operation.\n      updates: The values to be used in the operation.\n      name: the name of the operation.\n\n    Returns:\n      A `Tensor` that will hold the new value of this variable after\n      the scattered addition has completed.\n    '
        return gen_state_ops.scatter_nd_max(self._variable, indices, updates, use_locking=True, name=name)

    def scatter_nd_min(self, indices, updates, name=None):
        if False:
            while True:
                i = 10
        'Updates this variable with the min of `tf.IndexedSlices` and itself.\n\n    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.\n\n    `indices` must be integer tensor, containing indices into `ref`.\n    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.\n\n    The innermost dimension of `indices` (with length `K`) corresponds to\n    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th\n    dimension of `ref`.\n\n    `updates` is `Tensor` of rank `Q-1+P-K` with shape:\n\n    ```\n    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].\n    ```\n\n    See `tf.scatter_nd` for more details about how to make updates to\n    slices.\n\n    Args:\n      indices: The indices to be used in the operation.\n      updates: The values to be used in the operation.\n      name: the name of the operation.\n\n    Returns:\n      A `Tensor` that will hold the new value of this variable after\n      the scattered addition has completed.\n    '
        return gen_state_ops.scatter_nd_min(self._variable, indices, updates, use_locking=True, name=name)

    def _strided_slice_assign(self, begin, end, strides, value, name, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask):
        if False:
            for i in range(10):
                print('nop')
        return gen_array_ops.strided_slice_assign(ref=self._ref(), begin=begin, end=end, strides=strides, value=value, name=name, begin_mask=begin_mask, end_mask=end_mask, ellipsis_mask=ellipsis_mask, new_axis_mask=new_axis_mask, shrink_axis_mask=shrink_axis_mask)

    @deprecated(None, 'Prefer Dataset.range instead.')
    def count_up_to(self, limit):
        if False:
            i = 10
            return i + 15
        'Increments this variable until it reaches `limit`.\n\n    When that Op is run it tries to increment the variable by `1`. If\n    incrementing the variable would bring it above `limit` then the Op raises\n    the exception `OutOfRangeError`.\n\n    If no error is raised, the Op outputs the value of the variable before\n    the increment.\n\n    This is essentially a shortcut for `count_up_to(self, limit)`.\n\n    Args:\n      limit: value at which incrementing the variable raises an error.\n\n    Returns:\n      A `Tensor` that will hold the variable value before the increment. If no\n      other Op modifies this variable, the values produced will all be\n      distinct.\n    '
        return state_ops.count_up_to(self._variable, limit=limit)

    @staticmethod
    def _TensorConversionFunction(v, dtype=None, name=None, as_ref=False):
        if False:
            i = 10
            return i + 15
        'Utility function for converting a Variable to a Tensor.'
        _ = name
        if dtype and (not dtype.is_compatible_with(v.dtype)):
            raise ValueError("Incompatible type conversion requested to type '%s' for variable of type '%s'" % (dtype.name, v.dtype.name))
        if as_ref:
            return v._ref()
        else:
            return v.value()
    __array_priority__ = 100

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        'The name of this variable.'
        return self._name

    @property
    def initializer(self) -> ops.Operation:
        if False:
            i = 10
            return i + 15
        'The initializer operation for this variable.'
        return self._initializer_op

    @property
    def device(self):
        if False:
            i = 10
            return i + 15
        'The device of this variable.'
        return self._variable.device

    @property
    def dtype(self) -> dtypes.DType:
        if False:
            i = 10
            return i + 15
        'The `DType` of this variable.'
        return self._variable.dtype

    @property
    def op(self) -> ops.Operation:
        if False:
            return 10
        'The `Operation` of this variable.'
        return self._variable.op

    @property
    def graph(self) -> ops.Graph:
        if False:
            i = 10
            return i + 15
        'The `Graph` of this variable.'
        return self._variable.graph

    @property
    def _distribute_strategy(self):
        if False:
            while True:
                i = 10
        'The `tf.distribute.Strategy` that this variable was created under.'
        return None

    @property
    def shape(self):
        if False:
            print('Hello World!')
        'The `TensorShape` of this variable.\n\n    Returns:\n      A `TensorShape`.\n    '
        return self._variable.get_shape()

    def to_proto(self, export_scope=None):
        if False:
            i = 10
            return i + 15
        'Converts a `Variable` to a `VariableDef` protocol buffer.\n\n    Args:\n      export_scope: Optional `string`. Name scope to remove.\n\n    Returns:\n      A `VariableDef` protocol buffer, or `None` if the `Variable` is not\n      in the specified name scope.\n    '
        if export_scope is None or self._variable.name.startswith(export_scope):
            var_def = variable_pb2.VariableDef()
            var_def.variable_name = ops.strip_name_scope(self._variable.name, export_scope)
            if self._initial_value is not None:
                var_def.initial_value_name = ops.strip_name_scope(self._initial_value.name, export_scope)
            var_def.trainable = self.trainable
            var_def.synchronization = self.synchronization.value
            var_def.aggregation = self.aggregation.value
            var_def.initializer_name = ops.strip_name_scope(self.initializer.name, export_scope)
            var_def.snapshot_name = ops.strip_name_scope(self._snapshot.name, export_scope)
            if self._save_slice_info:
                var_def.save_slice_info_def.MergeFrom(self._save_slice_info.to_proto(export_scope=export_scope))
            return var_def
        else:
            return None

    def __iadd__(self, other):
        if False:
            for i in range(10):
                print('nop')
        logging.log_first_n(logging.WARN, "Variable += will be deprecated. Use variable.assign_add if you want assignment to the variable value or 'x = x + y' if you want a new python Tensor object.", 1)
        return self + other

    def __isub__(self, other):
        if False:
            while True:
                i = 10
        logging.log_first_n(logging.WARN, "Variable -= will be deprecated. Use variable.assign_sub if you want assignment to the variable value or 'x = x - y' if you want a new python Tensor object.", 1)
        return self - other

    def __imul__(self, other):
        if False:
            print('Hello World!')
        logging.log_first_n(logging.WARN, 'Variable *= will be deprecated. Use `var.assign(var * other)` if you want assignment to the variable value or `x = x * y` if you want a new python Tensor object.', 1)
        return self * other

    def __idiv__(self, other):
        if False:
            print('Hello World!')
        logging.log_first_n(logging.WARN, 'Variable /= will be deprecated. Use `var.assign(var / other)` if you want assignment to the variable value or `x = x / y` if you want a new python Tensor object.', 1)
        return self / other

    def __itruediv__(self, other):
        if False:
            return 10
        logging.log_first_n(logging.WARN, 'Variable /= will be deprecated. Use `var.assign(var / other)` if you want assignment to the variable value or `x = x / y` if you want a new python Tensor object.', 1)
        return self / other

    def __irealdiv__(self, other):
        if False:
            return 10
        logging.log_first_n(logging.WARN, 'Variable /= will be deprecated. Use `var.assign(var / other)` if you want assignment to the variable value or `x = x / y` if you want a new python Tensor object.', 1)
        return self / other

    def __ipow__(self, other):
        if False:
            return 10
        logging.log_first_n(logging.WARN, 'Variable **= will be deprecated. Use `var.assign(var ** other)` if you want assignment to the variable value or `x = x ** y` if you want a new python Tensor object.', 1)
        return self ** other

    def _serialize_to_tensors(self):
        if False:
            for i in range(10):
                print('nop')
        'Implements Trackable._serialize_to_tensors.'
        return {trackable.VARIABLE_VALUE_KEY: self}

    def _restore_from_tensors(self, restored_tensors):
        if False:
            while True:
                i = 10
        'Implements Trackable._restore_from_tensors.'
        restored_tensor = restored_tensors[trackable.VARIABLE_VALUE_KEY]
        return state_ops.assign(self, restored_tensor, validate_shape=self.get_shape().is_fully_defined())
tensor_conversion_registry.register_tensor_conversion_function(RefVariable, RefVariable._TensorConversionFunction)
variable_v1.set_variable_from_proto_fn(RefVariable)