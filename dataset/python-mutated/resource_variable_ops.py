"""Ops to use variables as resources."""
import contextlib
import functools
import weakref
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.checkpoint import tensor_callable
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.compat import compat as forward_compat
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.eager import tape
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_module
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.gen_resource_variable_ops import *
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
acd.register_read_only_resource_op('ReadVariableOp')
acd.register_read_only_resource_op('VariableShape')
acd.register_read_only_resource_op('ResourceGather')
acd.register_read_only_resource_op('ResourceGatherNd')
acd.register_read_only_resource_op('_ReadVariablesOp')
get_resource_handle_data = handle_data_util.get_resource_handle_data

def get_eager_safe_handle_data(handle):
    if False:
        i = 10
        return i + 15
    'Get the data handle from the Tensor `handle`.'
    assert isinstance(handle, tensor_module.Tensor)
    if isinstance(handle, ops.EagerTensor):
        return handle._handle_data
    else:
        return get_resource_handle_data(handle)

def _set_handle_shapes_and_types(tensor, handle_data, graph_mode):
    if False:
        for i in range(10):
            print('nop')
    'Sets the shape inference result HandleData on tensor.\n\n  Args:\n    tensor: A `Tensor` or `EagerTensor`.\n    handle_data: A `CppShapeInferenceResult.HandleData`.\n    graph_mode: A python bool.\n  '
    tensor._handle_data = handle_data
    if not graph_mode:
        return
    (shapes, types) = zip(*[(pair.shape, pair.dtype) for pair in handle_data.shape_and_type])
    ranks = [len(s.dim) if not s.unknown_rank else -1 for s in shapes]
    shapes = [[d.size for d in s.dim] if not s.unknown_rank else None for s in shapes]
    with tensor._op.graph._c_graph.get() as c_graph:
        pywrap_tf_session.TF_GraphSetOutputHandleShapesAndTypes_wrapper(c_graph, tensor._as_tf_output(), shapes, ranks, types)

def _combine_handle_data(handle, initial_value):
    if False:
        for i in range(10):
            print('nop')
    'Concats HandleData from tensors `handle` and `initial_value`.\n\n  Args:\n    handle: A `Tensor` of dtype `resource`.\n    initial_value: A `Tensor`.\n\n  Returns:\n    A `CppShapeInferenceResult.HandleData`.  If `initial_value` has dtype\n    `variant`, the `HandleData` contains the concatenation of the shape_and_type\n    from both `handle` and `initial_value`.\n\n  Raises:\n    RuntimeError: If handle, which was returned by VarHandleOp, either has\n      no handle data, or its len(handle_data.shape_and_type) != 1.\n  '
    assert handle.dtype == dtypes.resource
    variable_handle_data = get_eager_safe_handle_data(handle)
    if initial_value.dtype != dtypes.variant:
        return variable_handle_data
    extra_handle_data = get_eager_safe_handle_data(initial_value)
    if extra_handle_data is not None and extra_handle_data.is_set:
        if variable_handle_data is None or not variable_handle_data.is_set or len(variable_handle_data.shape_and_type) != 1:
            raise RuntimeError(f"Expected VarHandleOp to return a length==1 shape_and_type, but saw: '{variable_handle_data}'")
        variable_handle_data.shape_and_type.extend(extra_handle_data.shape_and_type)
    return variable_handle_data

def _variable_handle_from_shape_and_dtype(shape, dtype, shared_name, name, graph_mode, initial_value=None):
    if False:
        print('Hello World!')
    'Create a variable handle, copying in handle data from `initial_value`.'
    container = ops.get_default_graph()._container
    if container is None:
        container = ''
    shape = tensor_shape.as_shape(shape)
    dtype = dtypes.as_dtype(dtype)
    if not graph_mode:
        if shared_name is not None:
            raise errors.InternalError(node_def=None, op=None, message='Using an explicit shared_name is not allowed when executing eagerly.')
        shared_name = context.anonymous_name()
    handle = gen_resource_variable_ops.var_handle_op(shape=shape, dtype=dtype, shared_name=shared_name, debug_name=name, name=name, container=container)
    if initial_value is None:
        initial_value = handle
    if graph_mode:
        full_handle_data = _combine_handle_data(handle, initial_value)
        _set_handle_shapes_and_types(handle, full_handle_data, graph_mode)
        return handle
    else:
        handle_data = handle_data_util.create_handle_data(shape, dtype)
        if initial_value is not None and initial_value.dtype == dtypes.variant:
            extra_handle_data = get_eager_safe_handle_data(initial_value)
            if extra_handle_data is not None and extra_handle_data.is_set:
                if not handle_data.is_set or len(handle_data.shape_and_type) != 1:
                    raise RuntimeError(f"Expected VarHandleOp to return a length==1 shape_and_type, but saw: '{handle_data}'")
                handle_data.shape_and_type.extend(extra_handle_data.shape_and_type)
        _set_handle_shapes_and_types(handle, handle_data, graph_mode)
        return handle

def eager_safe_variable_handle(initial_value, shape, shared_name, name, graph_mode):
    if False:
        i = 10
        return i + 15
    "Creates a variable handle with information to do shape inference.\n\n  The dtype is read from `initial_value` and stored in the returned\n  resource tensor's handle data.\n\n  If `initial_value.dtype == tf.variant`, we additionally extract the handle\n  data (if any) from `initial_value` and append it to the `handle_data`.\n  In this case, the returned tensor's handle data is in the form\n\n  ```\n  is_set: true\n  shape_and_type {\n    shape {\n      // initial_value.shape\n    }\n    dtype: DT_VARIANT\n  }\n  shape_and_type {\n    // handle_data(initial_value).shape_and_type[0]\n  }\n  shape_and_type {\n    // handle_data(initial_value).shape_and_type[1]\n  }\n  ...\n  ```\n\n  Ops that read from this tensor, such as `ReadVariableOp` and\n  `AssignVariableOp`, know that `handle_data(handle).shape_and_type[1:]`\n  correspond to the handle data of the variant(s) stored in the Variable.\n\n  Args:\n    initial_value: A `Tensor`.\n    shape: The shape of the handle data. Can be `TensorShape(None)` (i.e.\n      unknown shape).\n    shared_name: A string.\n    name: A string.\n    graph_mode: A python bool.\n\n  Returns:\n    The handle, a `Tensor` of type `resource`.\n  "
    dtype = initial_value.dtype.base_dtype
    return _variable_handle_from_shape_and_dtype(shape, dtype, shared_name, name, graph_mode, initial_value)

@contextlib.contextmanager
def _handle_graph(handle):
    if False:
        i = 10
        return i + 15
    if context.executing_eagerly() or isinstance(handle, ops.EagerTensor) or ops.has_default_graph():
        yield
    else:
        with handle.graph.as_default():
            yield

class EagerResourceDeleter:
    """An object which cleans up a resource handle.

  An alternative to defining a __del__ method on an object. The intended use is
  that ResourceVariables or other objects with resource handles will maintain a
  single reference to this object. When the parent object is collected, this
  object will be too. Even if the parent object is part of a reference cycle,
  the cycle will be collectable.
  """
    __slots__ = ['_handle', '_handle_device', '_context']

    def __init__(self, handle, handle_device):
        if False:
            return 10
        if not isinstance(handle, tensor_module.Tensor):
            raise ValueError(f'Passed handle={handle} to EagerResourceDeleter. Was expecting the handle to be a `tf.Tensor`.')
        self._handle = handle
        self._handle_device = handle_device
        self._context = context.context()

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            if isinstance(self._handle, ops.EagerTensor) and self._handle.is_packed:
                return
            with context.eager_mode():
                with ops.device(self._handle_device):
                    gen_resource_variable_ops.destroy_resource_op(self._handle, ignore_lookup_error=True)
        except TypeError:
            pass
        except AttributeError:
            pass

def shape_safe_assign_variable_handle(handle, shape, value, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Helper that checks shape compatibility and assigns variable.'
    with _handle_graph(handle):
        value_tensor = ops.convert_to_tensor(value)
    shape.assert_is_compatible_with(value_tensor.shape)
    return gen_resource_variable_ops.assign_variable_op(handle, value_tensor, name=name)

def _maybe_set_handle_data(dtype, handle, tensor):
    if False:
        return 10
    if dtype == dtypes.variant:
        handle_data = get_eager_safe_handle_data(handle)
        if handle_data.is_set and len(handle_data.shape_and_type) > 1:
            tensor._handle_data = cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData(is_set=True, shape_and_type=handle_data.shape_and_type[1:])

def variable_accessed(variable):
    if False:
        print('Hello World!')
    'Records that `variable` was accessed for the tape and FuncGraph.'
    if hasattr(ops.get_default_graph(), 'watch_variable'):
        ops.get_default_graph().watch_variable(variable)
    if variable.trainable:
        tape.variable_accessed(variable)

def default_variable_creator_v2(next_creator=None, **kwargs):
    if False:
        while True:
            i = 10
    'Default variable creator.'
    assert next_creator is None
    initial_value = kwargs.get('initial_value', None)
    trainable = kwargs.get('trainable', None)
    validate_shape = kwargs.get('validate_shape', True)
    caching_device = kwargs.get('caching_device', None)
    name = kwargs.get('name', None)
    variable_def = kwargs.get('variable_def', None)
    dtype = kwargs.get('dtype', None)
    import_scope = kwargs.get('import_scope', None)
    constraint = kwargs.get('constraint', None)
    distribute_strategy = kwargs.get('distribute_strategy', None)
    synchronization = kwargs.get('synchronization', None)
    aggregation = kwargs.get('aggregation', None)
    shape = kwargs.get('shape', None)
    experimental_enable_variable_lifting = kwargs.get('experimental_enable_variable_lifting', None)
    return ResourceVariable(initial_value=initial_value, trainable=trainable, validate_shape=validate_shape, caching_device=caching_device, name=name, dtype=dtype, constraint=constraint, variable_def=variable_def, import_scope=import_scope, distribute_strategy=distribute_strategy, synchronization=synchronization, aggregation=aggregation, shape=shape, experimental_enable_variable_lifting=experimental_enable_variable_lifting)
variables.default_variable_creator_v2 = default_variable_creator_v2

class BaseResourceVariable(variables.Variable, core.Tensor):
    """A python variable from an existing handle."""

    def __init__(self, trainable=None, shape=None, dtype=None, handle=None, constraint=None, synchronization=None, aggregation=None, distribute_strategy=None, name=None, unique_id=None, handle_name=None, graph_element=None, initial_value=None, initializer_op=None, is_initialized_op=None, cached_value=None, save_slice_info=None, caching_device=None, in_graph_mode=None, validate_shape=True, **unused_kwargs):
        if False:
            return 10
        "Creates a variable from a handle.\n\n    Args:\n      trainable: If `True`, GradientTapes automatically watch uses of this\n        Variable.\n      shape: The variable's shape. This shape can be set to tf.TensorShape(None)\n        in order to assign values of different shapes to this variable.\n        Otherwise (i.e. if the shape is fully determined), it will trigger run\n        time checks to ensure that each assignment is of the same shape.\n      dtype: The variable's dtype.\n      handle: The variable's handle\n      constraint: An optional projection function to be applied to the variable\n        after being updated by an `Optimizer` (e.g. used to implement norm\n        constraints or value constraints for layer weights). The function must\n        take as input the unprojected Tensor representing the value of the\n        variable and return the Tensor for the projected value (which must have\n        the same shape). Constraints are not safe to use when doing asynchronous\n        distributed training.\n      synchronization: Indicates when a distributed a variable will be\n        aggregated. Accepted values are constants defined in the class\n        `tf.VariableSynchronization`. By default the synchronization is set to\n        `AUTO` and the current `DistributionStrategy` chooses when to\n        synchronize.\n      aggregation: Indicates how a distributed variable will be aggregated.\n        Accepted values are constants defined in the class\n        `tf.VariableAggregation`.\n      distribute_strategy: The distribution strategy this variable was created\n        under.\n      name: The name for this variable.\n      unique_id: Internal. Unique ID for this variable's handle.\n      handle_name: The name for the variable's handle.\n      graph_element: Optional, required only in session.run-mode. Pre-created\n        tensor which reads this variable's value.\n      initial_value: Optional. Variable's initial value.\n      initializer_op: Operation which assigns the variable's initial value.\n      is_initialized_op: Pre-created operation to check whether this variable is\n        initialized.\n      cached_value: Pre-created operation to read this variable in a specific\n        device.\n      save_slice_info: Metadata for variable partitioning.\n      caching_device: Optional device string or function describing where the\n        Variable should be cached for reading.  Defaults to the Variable's\n        device.  If not `None`, caches on another device.  Typical use is to\n        cache on the device where the Ops using the Variable reside, to\n        deduplicate copying through `Switch` and other conditional statements.\n      in_graph_mode: whether we are executing in TF1 graph mode. If None, will\n        detect within the function. This is to avoid repeated init_scope()\n        conetxt entrances which can add up.\n      validate_shape: If `False`, allows the variable to be initialized with a\n        value of unknown shape. If `True`, the default, the shape of\n        `initial_value` must be known.\n    "
        if in_graph_mode is None:
            with ops.init_scope():
                self._in_graph_mode = not context.executing_eagerly()
        else:
            self._in_graph_mode = in_graph_mode
        (synchronization, aggregation, trainable) = variables.validate_synchronization_aggregation_trainable(synchronization, aggregation, trainable, name)
        self._trainable = trainable
        self._synchronization = synchronization
        self._aggregation = aggregation
        self._save_slice_info = save_slice_info
        self._initial_value = initial_value
        self._initializer_op = initializer_op
        self._is_initialized_op = is_initialized_op
        self._graph_element = graph_element
        self._caching_device = caching_device
        self._cached_value = cached_value
        self._distribute_strategy = distribute_strategy
        self._graph_key = ops.get_default_graph()._graph_key
        self._shape = tensor_shape.as_shape(shape)
        self._dtype = dtypes.as_dtype(dtype)
        self._handle = handle
        self._unique_id = unique_id
        if handle_name is None:
            self._handle_name = 'Variable:0'
        else:
            self._handle_name = handle_name + ':0'
        self._constraint = constraint
        self._cached_shape_as_list = None
        self._validate_shape = validate_shape

    def __repr__(self):
        if False:
            print('Hello World!')
        if context.executing_eagerly() and (not self._in_graph_mode):
            try:
                with ops.device(self.device):
                    value_text = ops.value_text(self.read_value(), is_repr=True)
            except:
                value_text = 'numpy=<unavailable>'
            return "<tf.Variable '%s' shape=%s dtype=%s, %s>" % (self.name, self.get_shape(), self.dtype.name, value_text)
        else:
            return "<tf.Variable '%s' shape=%s dtype=%s>" % (self.name, self.get_shape(), self.dtype.name)

    def __tf_tracing_type__(self, signature_context):
        if False:
            while True:
                i = 10
        alias_id = signature_context.alias_global_id(self._handle._id)
        signature_context.add_placeholder(alias_id, self)
        return VariableSpec(shape=self.shape, dtype=self.dtype, trainable=self.trainable, alias_id=alias_id)

    @contextlib.contextmanager
    def _assign_dependencies(self):
        if False:
            i = 10
            return i + 15
        'Makes assignments depend on the cached value, if any.\n\n    This prevents undefined behavior with reads not ordered wrt writes.\n\n    Yields:\n      None.\n    '
        if self._cached_value is not None:
            with ops.control_dependencies([self._cached_value]):
                yield
        else:
            yield

    def __array__(self, dtype=None):
        if False:
            print('Hello World!')
        'Allows direct conversion to a numpy array.\n\n    >>> np.array(tf.Variable([1.0]))\n    array([1.], dtype=float32)\n\n    Returns:\n      The variable value as a numpy array.\n    '
        return np.asarray(self.numpy(), dtype=dtype)

    def __nonzero__(self):
        if False:
            i = 10
            return i + 15
        return self.__bool__()

    def __bool__(self):
        if False:
            print('Hello World!')
        return bool(self.read_value())

    def __copy__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __deepcopy__(self, memo):
        if False:
            i = 10
            return i + 15
        if not context.executing_eagerly():
            raise NotImplementedError('__deepcopy__() is only available when eager execution is enabled.')
        copied_variable = ResourceVariable(initial_value=self.read_value(), trainable=self._trainable, constraint=self._constraint, dtype=self._dtype, name=self._shared_name, distribute_strategy=self._distribute_strategy, synchronization=self.synchronization, aggregation=self.aggregation)
        memo[self._unique_id] = copied_variable
        return copied_variable

    @property
    def dtype(self):
        if False:
            while True:
                i = 10
        'The dtype of this variable.'
        return self._dtype

    @property
    def device(self):
        if False:
            return 10
        'The device this variable is on.'
        return self.handle.device

    @property
    def graph(self):
        if False:
            for i in range(10):
                print('nop')
        'The `Graph` of this variable.'
        return self.handle.graph

    @property
    def name(self):
        if False:
            print('Hello World!')
        'The name of the handle for this variable.'
        return self._handle_name

    @property
    def shape(self):
        if False:
            for i in range(10):
                print('nop')
        'The shape of this variable.'
        return self._shape

    def set_shape(self, shape):
        if False:
            while True:
                i = 10
        self._shape = self._shape.merge_with(shape)

    def _shape_as_list(self):
        if False:
            while True:
                i = 10
        if self.shape.ndims is None:
            return None
        return [dim.value for dim in self.shape.dims]

    def _shape_tuple(self):
        if False:
            for i in range(10):
                print('nop')
        shape = self._shape_as_list()
        if shape is None:
            return None
        return tuple(shape)

    @property
    def create(self):
        if False:
            for i in range(10):
                print('nop')
        'The op responsible for initializing this variable.'
        if not self._in_graph_mode:
            raise RuntimeError('This operation is not supported when eager execution is enabled.')
        return self._initializer_op

    @property
    def handle(self):
        if False:
            while True:
                i = 10
        'The handle by which this variable can be accessed.'
        return self._handle

    def value(self):
        if False:
            return 10
        'A cached operation which reads the value of this variable.'
        if self._cached_value is not None:
            return self._cached_value
        with ops.colocate_with(None, ignore_existing=True):
            return self._read_variable_op()

    def _as_graph_element(self):
        if False:
            print('Hello World!')
        'Conversion function for Graph.as_graph_element().'
        return self._graph_element

    @property
    def initializer(self):
        if False:
            while True:
                i = 10
        'The op responsible for initializing this variable.'
        return self._initializer_op

    @property
    def initial_value(self):
        if False:
            i = 10
            return i + 15
        'Returns the Tensor used as the initial value for the variable.'
        if context.executing_eagerly():
            raise RuntimeError('This property is not supported when eager execution is enabled.')
        return self._initial_value

    @property
    def constraint(self):
        if False:
            while True:
                i = 10
        'Returns the constraint function associated with this variable.\n\n    Returns:\n      The constraint function that was passed to the variable constructor.\n      Can be `None` if no constraint was passed.\n    '
        return self._constraint

    @property
    def op(self) -> ops.Operation:
        if False:
            return 10
        'The op for this variable.'
        return self.handle.op

    @property
    def trainable(self):
        if False:
            i = 10
            return i + 15
        return self._trainable

    @property
    def synchronization(self):
        if False:
            print('Hello World!')
        return self._synchronization

    @property
    def aggregation(self):
        if False:
            while True:
                i = 10
        return self._aggregation

    def eval(self, session=None):
        if False:
            for i in range(10):
                print('nop')
        'Evaluates and returns the value of this variable.'
        if context.executing_eagerly():
            raise RuntimeError('This operation is not supported when eager execution is enabled.')
        return self._graph_element.eval(session=session)

    def numpy(self):
        if False:
            print('Hello World!')
        if context.executing_eagerly():
            return self.read_value().numpy()
        raise NotImplementedError('numpy() is only available when eager execution is enabled.')

    @deprecated(None, 'Prefer Dataset.range instead.')
    def count_up_to(self, limit):
        if False:
            while True:
                i = 10
        'Increments this variable until it reaches `limit`.\n\n    When that Op is run it tries to increment the variable by `1`. If\n    incrementing the variable would bring it above `limit` then the Op raises\n    the exception `OutOfRangeError`.\n\n    If no error is raised, the Op outputs the value of the variable before\n    the increment.\n\n    This is essentially a shortcut for `count_up_to(self, limit)`.\n\n    Args:\n      limit: value at which incrementing the variable raises an error.\n\n    Returns:\n      A `Tensor` that will hold the variable value before the increment. If no\n      other Op modifies this variable, the values produced will all be\n      distinct.\n    '
        return gen_state_ops.resource_count_up_to(self.handle, limit=limit, T=self.dtype)

    def _copy_trackable_to_cpu(self, object_map):
        if False:
            print('Hello World!')
        'For implementing `Trackable`.'
        if self not in object_map:
            op_device = pydev.DeviceSpec.from_string(self.device).replace(device_type='CPU', device_index=0).to_string()
            with ops.device(op_device):
                new_var = UninitializedVariable(trainable=self.trainable, shape=self.shape, dtype=self.dtype, name=self._shared_name)
            object_map[self] = new_var
        destination_var = object_map[self]
        with ops.device(destination_var.device):
            destination_var.assign(self.read_value())

    def _export_to_saved_model_graph(self, object_map=None, tensor_map=None, options=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'For implementing `Trackable`.'
        new_variable = None
        if options.experimental_variable_policy._save_variable_devices():
            with ops.device(self.device):
                new_variable = copy_to_graph_uninitialized(self)
        else:
            new_variable = copy_to_graph_uninitialized(self)
        object_map[self] = new_variable
        tensor_map[self.handle] = new_variable.handle
        return [self.handle]

    def _serialize_to_tensors(self):
        if False:
            i = 10
            return i + 15
        'Implements Trackable._serialize_to_tensors.'

        def _read_variable_closure():
            if False:
                i = 10
                return i + 15
            v = self
            with ops.device(v.device):
                if context.executing_eagerly() and (not v.is_initialized()):
                    return None
                x = v.read_value_no_copy()
                with ops.device('/device:CPU:0'):
                    return array_ops.identity(x)
        return {trackable.VARIABLE_VALUE_KEY: tensor_callable.Callable(_read_variable_closure, dtype=self.dtype, device=self.device)}

    def _restore_from_tensors(self, restored_tensors):
        if False:
            i = 10
            return i + 15
        'Implements Trackable._restore_from_tensors.'
        with ops.device(self.device):
            restored_tensor = array_ops.identity(restored_tensors[trackable.VARIABLE_VALUE_KEY])
            try:
                assigned_variable = shape_safe_assign_variable_handle(self.handle, self.shape, restored_tensor)
            except ValueError as e:
                raise ValueError(f'Received incompatible tensor with shape {restored_tensor.shape} when attempting to restore variable with shape {self.shape} and name {self.name}.') from e
            return assigned_variable

    def _read_variable_op(self, no_copy=False):
        if False:
            for i in range(10):
                print('nop')
        'Reads the value of the variable.\n\n    If the variable is in copy-on-read mode and `no_copy` is True, the variable\n    is converted to copy-on-write mode before it is read.\n\n    Args:\n      no_copy: Whether to prevent a copy of the variable.\n\n    Returns:\n      The value of the variable.\n    '
        variable_accessed(self)

        def read_and_set_handle(no_copy):
            if False:
                print('Hello World!')
            if no_copy and forward_compat.forward_compatible(2022, 5, 3):
                gen_resource_variable_ops.disable_copy_on_read(self.handle)
            result = gen_resource_variable_ops.read_variable_op(self.handle, self._dtype)
            _maybe_set_handle_data(self._dtype, self.handle, result)
            return result
        if getattr(self, '_caching_device', None) is not None:
            with ops.colocate_with(None, ignore_existing=True):
                with ops.device(self._caching_device):
                    result = read_and_set_handle(no_copy)
        else:
            result = read_and_set_handle(no_copy)
        if not context.executing_eagerly():
            record.record_operation('ReadVariableOp', [result], [self.handle], backward_function=lambda x: [x], forward_function=lambda x: [x])
        return result

    def read_value(self):
        if False:
            return 10
        'Constructs an op which reads the value of this variable.\n\n    Should be used when there are multiple reads, or when it is desirable to\n    read the value only after some condition is true.\n\n    Returns:\n      The value of the variable.\n    '
        with ops.name_scope('Read'):
            value = self._read_variable_op()
        return array_ops.identity(value)

    def read_value_no_copy(self):
        if False:
            for i in range(10):
                print('nop')
        'Constructs an op which reads the value of this variable without copy.\n\n    The variable is read without making a copy even when it has been sparsely\n    accessed. Variables in copy-on-read mode will be converted to copy-on-write\n    mode.\n\n    Returns:\n      The value of the variable.\n    '
        with ops.name_scope('Read'):
            value = self._read_variable_op(no_copy=True)
        return array_ops.identity(value)

    def sparse_read(self, indices, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Reads the value of this variable sparsely, using `gather`.'
        with ops.name_scope('Gather' if name is None else name) as name:
            variable_accessed(self)
            value = gen_resource_variable_ops.resource_gather(self.handle, indices, dtype=self._dtype, name=name)
            if self._dtype == dtypes.variant:
                handle_data = get_eager_safe_handle_data(self.handle)
                if handle_data.is_set and len(handle_data.shape_and_type) > 1:
                    value._handle_data = cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData(is_set=True, shape_and_type=handle_data.shape_and_type[1:])
                return array_ops.identity(value)
        return value

    def gather_nd(self, indices, name=None):
        if False:
            while True:
                i = 10
        'Reads the value of this variable sparsely, using `gather_nd`.'
        with ops.name_scope('GatherNd' if name is None else name) as name:
            if self.trainable:
                variable_accessed(self)
            value = gen_resource_variable_ops.resource_gather_nd(self.handle, indices, dtype=self._dtype, name=name)
        return array_ops.identity(value)

    def to_proto(self, export_scope=None):
        if False:
            for i in range(10):
                print('nop')
        'Converts a `ResourceVariable` to a `VariableDef` protocol buffer.\n\n    Args:\n      export_scope: Optional `string`. Name scope to remove.\n\n    Raises:\n      RuntimeError: If run in EAGER mode.\n\n    Returns:\n      A `VariableDef` protocol buffer, or `None` if the `Variable` is not\n      in the specified name scope.\n    '
        if context.executing_eagerly():
            raise RuntimeError('This operation is not supported when eager execution is enabled.')
        if export_scope is None or self.handle.name.startswith(export_scope):
            var_def = variable_pb2.VariableDef()
            var_def.variable_name = ops.strip_name_scope(self.handle.name, export_scope)
            if self._initial_value is not None:
                var_def.initial_value_name = ops.strip_name_scope(self._initial_value.name, export_scope)
            var_def.initializer_name = ops.strip_name_scope(self.initializer.name, export_scope)
            if self._cached_value is not None:
                var_def.snapshot_name = ops.strip_name_scope(self._cached_value.name, export_scope)
            else:
                var_def.snapshot_name = ops.strip_name_scope(self._graph_element.name, export_scope)
            var_def.is_resource = True
            var_def.trainable = self.trainable
            var_def.synchronization = self.synchronization.value
            var_def.aggregation = self.aggregation.value
            if self._save_slice_info:
                var_def.save_slice_info_def.MergeFrom(self._save_slice_info.to_proto(export_scope=export_scope))
            return var_def
        else:
            return None

    @staticmethod
    def from_proto(variable_def, import_scope=None):
        if False:
            return 10
        if context.executing_eagerly():
            raise RuntimeError('This operation is not supported when eager execution is enabled.')
        return ResourceVariable(variable_def=variable_def, import_scope=import_scope)
    __array_priority__ = 100

    def is_initialized(self, name=None):
        if False:
            return 10
        'Checks whether a resource variable has been initialized.\n\n    Outputs boolean scalar indicating whether the tensor has been initialized.\n\n    Args:\n      name: A name for the operation (optional).\n\n    Returns:\n      A `Tensor` of type `bool`.\n    '
        return gen_resource_variable_ops.var_is_initialized_op(self.handle, name)

    def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
        if False:
            return 10
        'Subtracts a value from this variable.\n\n    Args:\n      delta: A `Tensor`. The value to subtract from this variable.\n      use_locking: If `True`, use locking during the operation.\n      name: The name to use for the operation.\n      read_value: A `bool`. Whether to read and return the new value of the\n        variable or not.\n\n    Returns:\n      If `read_value` is `True`, this method will return the new value of the\n      variable after the assignment has completed. Otherwise, when in graph mode\n      it will return the `Operation` that does the assignment, and when in eager\n      mode it will return `None`.\n    '
        with _handle_graph(self.handle), self._assign_dependencies():
            assign_sub_op = gen_resource_variable_ops.assign_sub_variable_op(self.handle, ops.convert_to_tensor(delta, dtype=self.dtype), name=name)
        if read_value:
            return self._lazy_read(assign_sub_op)
        return assign_sub_op

    def assign_add(self, delta, use_locking=None, name=None, read_value=True):
        if False:
            i = 10
            return i + 15
        'Adds a value to this variable.\n\n    Args:\n      delta: A `Tensor`. The value to add to this variable.\n      use_locking: If `True`, use locking during the operation.\n      name: The name to use for the operation.\n      read_value: A `bool`. Whether to read and return the new value of the\n        variable or not.\n\n    Returns:\n      If `read_value` is `True`, this method will return the new value of the\n      variable after the assignment has completed. Otherwise, when in graph mode\n      it will return the `Operation` that does the assignment, and when in eager\n      mode it will return `None`.\n    '
        with _handle_graph(self.handle), self._assign_dependencies():
            assign_add_op = gen_resource_variable_ops.assign_add_variable_op(self.handle, ops.convert_to_tensor(delta, dtype=self.dtype), name=name)
        if read_value:
            return self._lazy_read(assign_add_op)
        return assign_add_op

    def _lazy_read(self, op):
        if False:
            for i in range(10):
                print('nop')
        variable_accessed(self)
        return _UnreadVariable(handle=self.handle, dtype=self.dtype, shape=self._shape, in_graph_mode=self._in_graph_mode, parent_op=op, unique_id=self._unique_id)

    def assign(self, value, use_locking=None, name=None, read_value=True):
        if False:
            print('Hello World!')
        'Assigns a new value to this variable.\n\n    Args:\n      value: A `Tensor`. The new value for this variable.\n      use_locking: If `True`, use locking during the assignment.\n      name: The name to use for the assignment.\n      read_value: A `bool`. Whether to read and return the new value of the\n        variable or not.\n\n    Returns:\n      If `read_value` is `True`, this method will return the new value of the\n      variable after the assignment has completed. Otherwise, when in graph mode\n      it will return the `Operation` that does the assignment, and when in eager\n      mode it will return `None`.\n    '
        with _handle_graph(self.handle):
            value_tensor = ops.convert_to_tensor(value, dtype=self.dtype)
            if not self._shape.is_compatible_with(value_tensor.shape):
                if self.name is None:
                    tensor_name = ''
                else:
                    tensor_name = ' ' + str(self.name)
                raise ValueError(f"Cannot assign value to variable '{tensor_name}': Shape mismatch.The variable shape {self._shape}, and the assigned value shape {value_tensor.shape} are incompatible.")
            kwargs = {}
            if forward_compat.forward_compatible(2022, 3, 23):
                validate_shape = self._validate_shape and self._shape.is_fully_defined()
                kwargs['validate_shape'] = validate_shape
            assign_op = gen_resource_variable_ops.assign_variable_op(self.handle, value_tensor, name=name, **kwargs)
            if read_value:
                return self._lazy_read(assign_op)
        return assign_op

    def __reduce__(self):
        if False:
            while True:
                i = 10
        return (functools.partial(ResourceVariable, initial_value=self.numpy(), trainable=self.trainable, name=self._shared_name, dtype=self.dtype, constraint=self.constraint, distribute_strategy=self._distribute_strategy), ())

    def scatter_sub(self, sparse_delta, use_locking=False, name=None):
        if False:
            return 10
        'Subtracts `tf.IndexedSlices` from this variable.\n\n    Args:\n      sparse_delta: `tf.IndexedSlices` to be subtracted from this variable.\n      use_locking: If `True`, use locking during the operation.\n      name: the name of the operation.\n\n    Returns:\n      The updated variable.\n\n    Raises:\n      TypeError: if `sparse_delta` is not an `IndexedSlices`.\n    '
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError(f'Argument `sparse_delta` must be a `tf.IndexedSlices`. Received arg: {sparse_delta}')
        return self._lazy_read(gen_resource_variable_ops.resource_scatter_sub(self.handle, sparse_delta.indices, ops.convert_to_tensor(sparse_delta.values, self.dtype), name=name))

    def scatter_add(self, sparse_delta, use_locking=False, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Adds `tf.IndexedSlices` to this variable.\n\n    Args:\n      sparse_delta: `tf.IndexedSlices` to be added to this variable.\n      use_locking: If `True`, use locking during the operation.\n      name: the name of the operation.\n\n    Returns:\n      The updated variable.\n\n    Raises:\n      TypeError: if `sparse_delta` is not an `IndexedSlices`.\n    '
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError(f'Argument `sparse_delta` must be a `tf.IndexedSlices`. Received arg: {sparse_delta}')
        return self._lazy_read(gen_resource_variable_ops.resource_scatter_add(self.handle, sparse_delta.indices, ops.convert_to_tensor(sparse_delta.values, self.dtype), name=name))

    def scatter_max(self, sparse_delta, use_locking=False, name=None):
        if False:
            print('Hello World!')
        'Updates this variable with the max of `tf.IndexedSlices` and itself.\n\n    Args:\n      sparse_delta: `tf.IndexedSlices` to use as an argument of max with this\n        variable.\n      use_locking: If `True`, use locking during the operation.\n      name: the name of the operation.\n\n    Returns:\n      The updated variable.\n\n    Raises:\n      TypeError: if `sparse_delta` is not an `IndexedSlices`.\n    '
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError(f'Argument `sparse_delta` must be a `tf.IndexedSlices`. Received arg: {sparse_delta}')
        return self._lazy_read(gen_resource_variable_ops.resource_scatter_max(self.handle, sparse_delta.indices, ops.convert_to_tensor(sparse_delta.values, self.dtype), name=name))

    def scatter_min(self, sparse_delta, use_locking=False, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Updates this variable with the min of `tf.IndexedSlices` and itself.\n\n    Args:\n      sparse_delta: `tf.IndexedSlices` to use as an argument of min with this\n        variable.\n      use_locking: If `True`, use locking during the operation.\n      name: the name of the operation.\n\n    Returns:\n      The updated variable.\n\n    Raises:\n      TypeError: if `sparse_delta` is not an `IndexedSlices`.\n    '
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError(f'Argument `sparse_delta` must be a `tf.IndexedSlices`. Received arg: {sparse_delta}')
        return self._lazy_read(gen_resource_variable_ops.resource_scatter_min(self.handle, sparse_delta.indices, ops.convert_to_tensor(sparse_delta.values, self.dtype), name=name))

    def scatter_mul(self, sparse_delta, use_locking=False, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Multiply this variable by `tf.IndexedSlices`.\n\n    Args:\n      sparse_delta: `tf.IndexedSlices` to multiply this variable by.\n      use_locking: If `True`, use locking during the operation.\n      name: the name of the operation.\n\n    Returns:\n      The updated variable.\n\n    Raises:\n      TypeError: if `sparse_delta` is not an `IndexedSlices`.\n    '
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError(f'Argument `sparse_delta` must be a `tf.IndexedSlices`. Received arg: {sparse_delta}')
        return self._lazy_read(gen_resource_variable_ops.resource_scatter_mul(self.handle, sparse_delta.indices, ops.convert_to_tensor(sparse_delta.values, self.dtype), name=name))

    def scatter_div(self, sparse_delta, use_locking=False, name=None):
        if False:
            return 10
        'Divide this variable by `tf.IndexedSlices`.\n\n    Args:\n      sparse_delta: `tf.IndexedSlices` to divide this variable by.\n      use_locking: If `True`, use locking during the operation.\n      name: the name of the operation.\n\n    Returns:\n      The updated variable.\n\n    Raises:\n      TypeError: if `sparse_delta` is not an `IndexedSlices`.\n    '
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError(f'Argument `sparse_delta` must be a `tf.IndexedSlices`. Received arg: {sparse_delta}')
        return self._lazy_read(gen_resource_variable_ops.resource_scatter_div(self.handle, sparse_delta.indices, ops.convert_to_tensor(sparse_delta.values, self.dtype), name=name))

    def scatter_update(self, sparse_delta, use_locking=False, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Assigns `tf.IndexedSlices` to this variable.\n\n    Args:\n      sparse_delta: `tf.IndexedSlices` to be assigned to this variable.\n      use_locking: If `True`, use locking during the operation.\n      name: the name of the operation.\n\n    Returns:\n      The updated variable.\n\n    Raises:\n      TypeError: if `sparse_delta` is not an `IndexedSlices`.\n    '
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError(f'Argument `sparse_delta` must be a `tf.IndexedSlices`. Received arg: {sparse_delta}')
        return self._lazy_read(gen_resource_variable_ops.resource_scatter_update(self.handle, sparse_delta.indices, ops.convert_to_tensor(sparse_delta.values, self.dtype), name=name))

    def batch_scatter_update(self, sparse_delta, use_locking=False, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Assigns `tf.IndexedSlices` to this variable batch-wise.\n\n    Analogous to `batch_gather`. This assumes that this variable and the\n    sparse_delta IndexedSlices have a series of leading dimensions that are the\n    same for all of them, and the updates are performed on the last dimension of\n    indices. In other words, the dimensions should be the following:\n\n    `num_prefix_dims = sparse_delta.indices.ndims - 1`\n    `batch_dim = num_prefix_dims + 1`\n    `sparse_delta.updates.shape = sparse_delta.indices.shape + var.shape[\n         batch_dim:]`\n\n    where\n\n    `sparse_delta.updates.shape[:num_prefix_dims]`\n    `== sparse_delta.indices.shape[:num_prefix_dims]`\n    `== var.shape[:num_prefix_dims]`\n\n    And the operation performed can be expressed as:\n\n    `var[i_1, ..., i_n,\n         sparse_delta.indices[i_1, ..., i_n, j]] = sparse_delta.updates[\n            i_1, ..., i_n, j]`\n\n    When sparse_delta.indices is a 1D tensor, this operation is equivalent to\n    `scatter_update`.\n\n    To avoid this operation one can looping over the first `ndims` of the\n    variable and using `scatter_update` on the subtensors that result of slicing\n    the first dimension. This is a valid option for `ndims = 1`, but less\n    efficient than this implementation.\n\n    Args:\n      sparse_delta: `tf.IndexedSlices` to be assigned to this variable.\n      use_locking: If `True`, use locking during the operation.\n      name: the name of the operation.\n\n    Returns:\n      The updated variable.\n\n    Raises:\n      TypeError: if `sparse_delta` is not an `IndexedSlices`.\n    '
        if not isinstance(sparse_delta, indexed_slices.IndexedSlices):
            raise TypeError(f'Argument `sparse_delta` must be a `tf.IndexedSlices`. Received arg: {sparse_delta}')
        return self._lazy_read(state_ops.batch_scatter_update(self, sparse_delta.indices, sparse_delta.values, use_locking=use_locking, name=name))

    def scatter_nd_sub(self, indices, updates, name=None):
        if False:
            return 10
        'Applies sparse subtraction to individual values or slices in a Variable.\n\n    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.\n\n    `indices` must be integer tensor, containing indices into `ref`.\n    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.\n\n    The innermost dimension of `indices` (with length `K`) corresponds to\n    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th\n    dimension of `ref`.\n\n    `updates` is `Tensor` of rank `Q-1+P-K` with shape:\n\n    ```\n    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].\n    ```\n\n    For example, say we want to add 4 scattered elements to a rank-1 tensor to\n    8 elements. In Python, that update would look like this:\n\n    ```python\n        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])\n        indices = tf.constant([[4], [3], [1] ,[7]])\n        updates = tf.constant([9, 10, 11, 12])\n        op = ref.scatter_nd_sub(indices, updates)\n        with tf.compat.v1.Session() as sess:\n          print sess.run(op)\n    ```\n\n    The resulting update to ref would look like this:\n\n        [1, -9, 3, -6, -6, 6, 7, -4]\n\n    See `tf.scatter_nd` for more details about how to make updates to\n    slices.\n\n    Args:\n      indices: The indices to be used in the operation.\n      updates: The values to be used in the operation.\n      name: the name of the operation.\n\n    Returns:\n      The updated variable.\n    '
        return self._lazy_read(gen_state_ops.resource_scatter_nd_sub(self.handle, indices, ops.convert_to_tensor(updates, self.dtype), name=name))

    def scatter_nd_add(self, indices, updates, name=None):
        if False:
            print('Hello World!')
        'Applies sparse addition to individual values or slices in a Variable.\n\n    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.\n\n    `indices` must be integer tensor, containing indices into `ref`.\n    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.\n\n    The innermost dimension of `indices` (with length `K`) corresponds to\n    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th\n    dimension of `ref`.\n\n    `updates` is `Tensor` of rank `Q-1+P-K` with shape:\n\n    ```\n    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].\n    ```\n\n    For example, say we want to add 4 scattered elements to a rank-1 tensor to\n    8 elements. In Python, that update would look like this:\n\n    ```python\n        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])\n        indices = tf.constant([[4], [3], [1] ,[7]])\n        updates = tf.constant([9, 10, 11, 12])\n        add = ref.scatter_nd_add(indices, updates)\n        with tf.compat.v1.Session() as sess:\n          print sess.run(add)\n    ```\n\n    The resulting update to ref would look like this:\n\n        [1, 13, 3, 14, 14, 6, 7, 20]\n\n    See `tf.scatter_nd` for more details about how to make updates to\n    slices.\n\n    Args:\n      indices: The indices to be used in the operation.\n      updates: The values to be used in the operation.\n      name: the name of the operation.\n\n    Returns:\n      The updated variable.\n    '
        return self._lazy_read(gen_state_ops.resource_scatter_nd_add(self.handle, indices, ops.convert_to_tensor(updates, self.dtype), name=name))

    def scatter_nd_update(self, indices, updates, name=None):
        if False:
            i = 10
            return i + 15
        'Applies sparse assignment to individual values or slices in a Variable.\n\n    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.\n\n    `indices` must be integer tensor, containing indices into `ref`.\n    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.\n\n    The innermost dimension of `indices` (with length `K`) corresponds to\n    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th\n    dimension of `ref`.\n\n    `updates` is `Tensor` of rank `Q-1+P-K` with shape:\n\n    ```\n    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].\n    ```\n\n    For example, say we want to add 4 scattered elements to a rank-1 tensor to\n    8 elements. In Python, that update would look like this:\n\n    ```python\n        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])\n        indices = tf.constant([[4], [3], [1] ,[7]])\n        updates = tf.constant([9, 10, 11, 12])\n        op = ref.scatter_nd_update(indices, updates)\n        with tf.compat.v1.Session() as sess:\n          print sess.run(op)\n    ```\n\n    The resulting update to ref would look like this:\n\n        [1, 11, 3, 10, 9, 6, 7, 12]\n\n    See `tf.scatter_nd` for more details about how to make updates to\n    slices.\n\n    Args:\n      indices: The indices to be used in the operation.\n      updates: The values to be used in the operation.\n      name: the name of the operation.\n\n    Returns:\n      The updated variable.\n    '
        return self._lazy_read(gen_state_ops.resource_scatter_nd_update(self.handle, indices, ops.convert_to_tensor(updates, self.dtype), name=name))

    def scatter_nd_max(self, indices, updates, name=None):
        if False:
            print('Hello World!')
        'Updates this variable with the max of `tf.IndexedSlices` and itself.\n\n    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.\n\n    `indices` must be integer tensor, containing indices into `ref`.\n    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.\n\n    The innermost dimension of `indices` (with length `K`) corresponds to\n    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th\n    dimension of `ref`.\n\n    `updates` is `Tensor` of rank `Q-1+P-K` with shape:\n\n    ```\n    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].\n    ```\n\n    See `tf.scatter_nd` for more details about how to make updates to\n    slices.\n\n    Args:\n      indices: The indices to be used in the operation.\n      updates: The values to be used in the operation.\n      name: the name of the operation.\n\n    Returns:\n      The updated variable.\n    '
        return self._lazy_read(gen_state_ops.resource_scatter_nd_max(self.handle, indices, ops.convert_to_tensor(updates, self.dtype), name=name))

    def scatter_nd_min(self, indices, updates, name=None):
        if False:
            while True:
                i = 10
        'Updates this variable with the min of `tf.IndexedSlices` and itself.\n\n    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.\n\n    `indices` must be integer tensor, containing indices into `ref`.\n    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.\n\n    The innermost dimension of `indices` (with length `K`) corresponds to\n    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th\n    dimension of `ref`.\n\n    `updates` is `Tensor` of rank `Q-1+P-K` with shape:\n\n    ```\n    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].\n    ```\n\n    See `tf.scatter_nd` for more details about how to make updates to\n    slices.\n\n    Args:\n      indices: The indices to be used in the operation.\n      updates: The values to be used in the operation.\n      name: the name of the operation.\n\n    Returns:\n      The updated variable.\n    '
        return self._lazy_read(gen_state_ops.resource_scatter_nd_min(self.handle, indices, ops.convert_to_tensor(updates, self.dtype), name=name))

    def _write_object_proto(self, proto, options):
        if False:
            print('Hello World!')
        'Writes additional information of the variable into the SavedObject proto.\n\n    Subclasses of ResourceVariables could choose to override this method to\n    customize extra information to provide when saving a SavedModel.\n\n    Ideally, this should contain the logic in\n    write_object_proto_for_resource_variable but `DistributedValue` is an\n    outlier at the momemnt. Once `DistributedValue` becomes a proper\n    ResourceVariable, we should remove the helper method below.\n\n    Args:\n      proto: `SavedObject` proto to update.\n      options: A `SaveOption` instance that configures save behavior.\n    '
        write_object_proto_for_resource_variable(self, proto, options)

    def _strided_slice_assign(self, begin, end, strides, value, name, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask):
        if False:
            print('Hello World!')
        with _handle_graph(self.handle), self._assign_dependencies():
            return self._lazy_read(gen_array_ops.resource_strided_slice_assign(ref=self.handle, begin=begin, end=end, strides=strides, value=ops.convert_to_tensor(value, dtype=self.dtype), name=name, begin_mask=begin_mask, end_mask=end_mask, ellipsis_mask=ellipsis_mask, new_axis_mask=new_axis_mask, shrink_axis_mask=shrink_axis_mask))

    def __complex__(self):
        if False:
            for i in range(10):
                print('nop')
        return complex(self.value().numpy())

    def __int__(self):
        if False:
            print('Hello World!')
        return int(self.value().numpy())

    def __long__(self):
        if False:
            while True:
                i = 10
        return long(self.value().numpy())

    def __float__(self):
        if False:
            print('Hello World!')
        return float(self.value().numpy())

    def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
        if False:
            return 10
        del name
        if dtype is not None and (not dtype.is_compatible_with(self.dtype)):
            raise ValueError(f'Incompatible type conversion requested to type {dtype.name} for `tf.Variable of type {self.dtype.name}. (Variable: {self})')
        if as_ref:
            return self.read_value().op.inputs[0]
        else:
            return self.value()

    def __iadd__(self, unused_other):
        if False:
            print('Hello World!')
        raise RuntimeError('`variable += value` with `tf.Variable`s is not supported. Use `variable.assign_add(value)` to modify the variable, or `out = variable + value` if you need to get a new output Tensor.')

    def __isub__(self, unused_other):
        if False:
            for i in range(10):
                print('nop')
        raise RuntimeError('`variable -= value` with `tf.Variable`s is not supported. Use `variable.assign_sub(value)` to modify the variable, or `out = variable * value` if you need to get a new output Tensor.')

    def __imul__(self, unused_other):
        if False:
            for i in range(10):
                print('nop')
        raise RuntimeError('`var *= value` with `tf.Variable`s is not supported. Use `var.assign(var * value)` to modify the variable, or `out = var * value` if you need to get a new output Tensor.')

    def __idiv__(self, unused_other):
        if False:
            i = 10
            return i + 15
        raise RuntimeError('`var /= value` with `tf.Variable`s is not supported. Use `var.assign(var / value)` to modify the variable, or `out = var / value` if you need to get a new output Tensor.')

    def __itruediv__(self, unused_other):
        if False:
            for i in range(10):
                print('nop')
        raise RuntimeError('`var /= value` with `tf.Variable`s is not supported. Use `var.assign(var / value)` to modify the variable, or `out = var / value` if you need to get a new output Tensor.')

    def __irealdiv__(self, unused_other):
        if False:
            print('Hello World!')
        raise RuntimeError('`var /= value` with `tf.Variable`s is not supported. Use `var.assign(var / value)` to modify the variable, or `out = var / value` if you need to get a new output Tensor.')

    def __ipow__(self, unused_other):
        if False:
            while True:
                i = 10
        raise RuntimeError('`var **= value` with `tf.Variable`s is not supported. Use `var.assign(var ** value)` to modify the variable, or `out = var ** value` if you need to get a new output Tensor.')

class ResourceVariableGradient(composite_tensor_gradient.CompositeTensorGradient):
    """CompositeTensorGradient protocol for ResourceVariable."""

    def get_gradient_components(self, value):
        if False:
            print('Hello World!')
        'Returns the components of `value` that should be included in gradients.\n\n    For a ResourceVariable, its gradient component is its handle tensor.\n    For now, we return the ResourceVariable because the gradient infrastructure\n    has special logics to handle ResourceVariables. We should remove those\n    special logics and return the handle tensor.\n\n    Args:\n      value: A `ResourceVariable`.\n\n    Returns:\n      `value` itself.\n    '
        return value

    def replace_gradient_components(self, value, component_grads):
        if False:
            i = 10
            return i + 15
        "Replaces the gradient components in `value` with `component_grads`.\n\n    The gradient of a ResourceVariable is either None or a Tensor. So we don't\n    need `value`'s TypeSpec or non-gradient components in this method.\n\n    Args:\n      value: A `ResourceVariable` with its gradient components compatible with\n        `component_grads`.\n      component_grads: A `Tensor` or None as the gradient result.\n\n    Returns:\n      The `component_grads`, which is either a `Tensor` or None.\n    "
        return component_grads

class ResourceVariable(BaseResourceVariable, composite_tensor.CompositeTensor):
    """Variable based on resource handles.

  See the [Variables How To](https://tensorflow.org/guide/variables)
  for a high level overview.

  A `ResourceVariable` allows you to maintain state across subsequent calls to
  session.run.

  The `ResourceVariable` constructor requires an initial value for the variable,
  which can be a `Tensor` of any type and shape. The initial value defines the
  type and shape of the variable. After construction, the type and shape of
  the variable are fixed. The value can be changed using one of the assign
  methods.

  Just like any `Tensor`, variables created with
  `tf.Variable(use_resource=True)` can be used as inputs for other Ops in the
  graph. Additionally, all the operators overloaded for the `Tensor` class are
  carried over to variables, so you can also add nodes to the graph by just
  doing arithmetic on variables.

  Unlike ref-based variable, a ResourceVariable has well-defined semantics. Each
  usage of a ResourceVariable in a TensorFlow graph adds a read_value operation
  to the graph. The Tensors returned by a read_value operation are guaranteed to
  see all modifications to the value of the variable which happen in any
  operation on which the read_value depends on (either directly, indirectly, or
  via a control dependency) and guaranteed to not see any modification to the
  value of the variable from operations that depend on the read_value operation.
  Updates from operations that have no dependency relationship to the read_value
  operation might or might not be visible to read_value.

  For example, if there is more than one assignment to a ResourceVariable in
  a single session.run call there is a well-defined value for each operation
  which uses the variable's value if the assignments and the read are connected
  by edges in the graph. Consider the following example, in which two writes
  can cause tf.Variable and tf.ResourceVariable to behave differently:

  ```python
  a = tf.Variable(1.0, use_resource=True)
  a.initializer.run()

  assign = a.assign(2.0)
  with tf.control_dependencies([assign]):
    b = a.read_value()
  with tf.control_dependencies([b]):
    other_assign = a.assign(3.0)
  with tf.control_dependencies([other_assign]):
    # Will print 2.0 because the value was read before other_assign ran. If
    # `a` was a tf.Variable instead, 2.0 or 3.0 could be printed.
    tf.compat.v1.Print(b, [b]).eval()
  ```
  """

    def __init__(self, initial_value=None, trainable=None, collections=None, validate_shape=True, caching_device=None, name=None, dtype=None, variable_def=None, import_scope=None, constraint=None, distribute_strategy=None, synchronization=None, aggregation=None, shape=None, handle=None, experimental_enable_variable_lifting=None):
        if False:
            return 10
        'Creates a variable.\n\n    Args:\n      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,\n        which is the initial value for the Variable. Can also be a callable with\n        no argument that returns the initial value when called. (Note that\n        initializer functions from init_ops.py must first be bound to a shape\n        before being used here.)\n      trainable: If `True`, the default, also adds the variable to the graph\n        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as\n        the default list of variables to use by the `Optimizer` classes.\n        Defaults to `True`, unless `synchronization` is set to `ON_READ`, in\n        which case it defaults to `False`.\n      collections: List of graph collections keys. The new variable is added to\n        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.\n      validate_shape: If `False`, allows the variable to be initialized with a\n        value of unknown shape. If `True`, the default, the shape of\n        `initial_value` must be known.\n      caching_device: Optional device string or function describing where the\n        Variable should be cached for reading.  Defaults to the Variable\'s\n        device.  If not `None`, caches on another device.  Typical use is to\n        cache on the device where the Ops using the Variable reside, to\n        deduplicate copying through `Switch` and other conditional statements.\n      name: Optional name for the variable. Defaults to `\'Variable\'` and gets\n        uniquified automatically.\n      dtype: If set, initial_value will be converted to the given type. If None,\n        either the datatype will be kept (if initial_value is a Tensor) or\n        float32 will be used (if it is a Python object convertible to a Tensor).\n      variable_def: `VariableDef` protocol buffer. If not None, recreates the\n        `ResourceVariable` object with its contents. `variable_def` and other\n        arguments (except for import_scope) are mutually exclusive.\n      import_scope: Optional `string`. Name scope to add to the\n        ResourceVariable. Only used when `variable_def` is provided.\n      constraint: An optional projection function to be applied to the variable\n        after being updated by an `Optimizer` (e.g. used to implement norm\n        constraints or value constraints for layer weights). The function must\n        take as input the unprojected Tensor representing the value of the\n        variable and return the Tensor for the projected value (which must have\n        the same shape). Constraints are not safe to use when doing asynchronous\n        distributed training.\n      distribute_strategy: The tf.distribute.Strategy this variable is being\n        created inside of.\n      synchronization: Indicates when a distributed a variable will be\n        aggregated. Accepted values are constants defined in the class\n        `tf.VariableSynchronization`. By default the synchronization is set to\n        `AUTO` and the current `DistributionStrategy` chooses when to\n        synchronize.\n      aggregation: Indicates how a distributed variable will be aggregated.\n        Accepted values are constants defined in the class\n        `tf.VariableAggregation`.\n      shape: (optional) The shape of this variable. If None, the shape of\n        `initial_value` will be used. When setting this argument to\n        `tf.TensorShape(None)` (representing an unspecified shape), the variable\n        can be assigned with values of different shapes.\n      handle: (optional) The handle of a `tf.Variable`. If provided, only\n        `trainable`, `shape`, `dtype`, and `handle` will be used to construct\n        this `tf.Variable`.\n      experimental_enable_variable_lifting: Whether to lift the variable out if\n        it\'s in a `tf.function`. Default is `True`. When this argument\n        is `True`, variable creation will follow the behavior and\n        restrictions described\n        [here](https://www.tensorflow.org/guide/function#creating_tfvariables).\n        If this argument is `False`, that description doesn\'t apply,\n        and you can freely create and use the variable in the\n        `tf.function`, as if it\'s a "mutable `tf.Tensor`". You can\'t\n        return the variable though.\n\n    Raises:\n      ValueError: If the initial value is not specified, or does not have a\n        shape and `validate_shape` is `True`.\n\n    @compatibility(eager)\n    When Eager Execution is enabled, the default for the `collections` argument\n    is `None`, which signifies that this `Variable` will not be added to any\n    collections.\n    @end_compatibility\n    '
        if variable_def:
            if initial_value is not None:
                raise ValueError(f'The variable_def and initial_value args to `tf.Variable` are mutually exclusive, but got both: variable_def={variable_def},\ninitial_value={initial_value}')
            if context.executing_eagerly():
                raise ValueError(f'Creating a `tf.Variable` with a `variable_def` arg is not supported when eager execution is enabled. Got: variable_def={variable_def}')
            self._init_from_proto(variable_def, import_scope=import_scope, validate_shape=validate_shape)
        elif handle is not None:
            self._init_from_handle(trainable=trainable, shape=shape, dtype=dtype, handle=handle)
        else:
            self._init_from_args(initial_value=initial_value, trainable=trainable, collections=collections, caching_device=caching_device, name=name, dtype=dtype, constraint=constraint, synchronization=synchronization, aggregation=aggregation, shape=shape, distribute_strategy=distribute_strategy, validate_shape=validate_shape, experimental_enable_variable_lifting=experimental_enable_variable_lifting)

    @property
    def _type_spec(self):
        if False:
            while True:
                i = 10
        return VariableSpec.from_value(self)

    def _shape_invariant_to_type_spec(self, shape):
        if False:
            while True:
                i = 10
        return VariableSpec(shape, self.dtype, self.trainable)
    __composite_gradient__ = ResourceVariableGradient()

    def _init_from_args(self, initial_value=None, trainable=None, collections=None, caching_device=None, name=None, dtype=None, constraint=None, synchronization=None, aggregation=None, distribute_strategy=None, shape=None, validate_shape=True, experimental_enable_variable_lifting=None):
        if False:
            i = 10
            return i + 15
        'Creates a variable.\n\n    Args:\n      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,\n        which is the initial value for the Variable. The initial value must have\n        a shape specified unless `validate_shape` is set to False. Can also be a\n        callable with no argument that returns the initial value when called.\n        (Note that initializer functions from init_ops.py must first be bound to\n        a shape before being used here.)\n      trainable: If `True`, the default, also adds the variable to the graph\n        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as\n        the default list of variables to use by the `Optimizer` classes.\n        Defaults to `True`, unless `synchronization` is set to `ON_READ`, in\n        which case it defaults to `False`.\n      collections: List of graph collections keys. The new variable is added to\n        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.\n      caching_device: Optional device string or function describing where the\n        Variable should be cached for reading.  Defaults to the Variable\'s\n        device.  If not `None`, caches on another device.  Typical use is to\n        cache on the device where the Ops using the Variable reside, to\n        deduplicate copying through `Switch` and other conditional statements.\n      name: Optional name for the variable. Defaults to `\'Variable\'` and gets\n        uniquified automatically.\n      dtype: If set, initial_value will be converted to the given type. If None,\n        either the datatype will be kept (if initial_value is a Tensor) or\n        float32 will be used (if it is a Python object convertible to a Tensor).\n      constraint: An optional projection function to be applied to the variable\n        after being updated by an `Optimizer` (e.g. used to implement norm\n        constraints or value constraints for layer weights). The function must\n        take as input the unprojected Tensor representing the value of the\n        variable and return the Tensor for the projected value (which must have\n        the same shape). Constraints are not safe to use when doing asynchronous\n        distributed training.\n      synchronization: Indicates when a distributed a variable will be\n        aggregated. Accepted values are constants defined in the class\n        `tf.VariableSynchronization`. By default the synchronization is set to\n        `AUTO` and the current `DistributionStrategy` chooses when to\n        synchronize.\n      aggregation: Indicates how a distributed variable will be aggregated.\n        Accepted values are constants defined in the class\n        `tf.VariableAggregation`.\n      distribute_strategy: DistributionStrategy under which this variable was\n        created.\n      shape: (optional) The shape of this variable. If None, the shape of\n        `initial_value` will be used. When setting this argument to\n        `tf.TensorShape(None)` (representing an unspecified shape), the variable\n        can be assigned with values of different shapes.\n      validate_shape: If `False`, allows the variable to be initialized with a\n        value of unknown shape. If `True`, the default, the shape of\n        `initial_value` must be known.\n      experimental_enable_variable_lifting: Whether to lift the variable out if\n        it\'s in a `tf.function`. Default is `True`. When this argument\n        is `True`, variable creation will follow the behavior and\n        restrictions described\n        [here](https://www.tensorflow.org/guide/function#creating_tfvariables).\n        If this argument is `False`, that description doesn\'t apply,\n        and you can freely create and use the variable in the\n        `tf.function`, as if it\'s a "mutable `tf.Tensor`". You can\'t\n        return the variable though.\n\n    Raises:\n      ValueError: If the initial value is not specified, or does not have a\n        shape and `validate_shape` is `True`.\n\n    @compatibility(eager)\n    When Eager Execution is enabled, variables are never added to collections.\n    It is not implicitly added to the `GLOBAL_VARIABLES` or\n    `TRAINABLE_VARIABLES` collections, and the `collections` argument is\n    ignored.\n    @end_compatibility\n    '
        (synchronization, aggregation, trainable) = variables.validate_synchronization_aggregation_trainable(synchronization, aggregation, trainable, name)
        if experimental_enable_variable_lifting is None:
            experimental_enable_variable_lifting = True
        if initial_value is None:
            raise ValueError('The `initial_value` arg to `tf.Variable` must be specified except when you are not providing a `variable_def`. You provided neither.')
        init_from_fn = callable(initial_value)
        if isinstance(initial_value, tensor_module.Tensor) and hasattr(initial_value, 'graph') and initial_value.graph.building_function:
            raise ValueError(f"Argument `initial_value` ({initial_value}) could not be lifted out of a `tf.function`. (Tried to create variable with name='{name}'). To avoid this error, when constructing `tf.Variable`s inside of `tf.function` you can create the `initial_value` tensor in a `tf.init_scope` or pass a callable `initial_value` (e.g., `tf.Variable(lambda : tf.truncated_normal([10, 40]))`). Please file a feature request if this restriction inconveniences you.")
        if collections is None:
            collections = [ops.GraphKeys.GLOBAL_VARIABLES]
        if not isinstance(collections, (list, tuple, set)):
            raise ValueError(f'collections argument to Variable constructor must be a list, tuple, or set. Got {collections} of type {type(collections)}')
        if constraint is not None and (not callable(constraint)):
            raise ValueError(f'Argument `constraint` must be None or a callable. a callable. Got a {type(constraint)}:  {constraint}')
        if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
            collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]
        with ops.init_scope():
            self._in_graph_mode = not context.executing_eagerly()
        if experimental_enable_variable_lifting:
            maybe_init_scope = ops.init_scope
        else:
            maybe_init_scope = contextlib.nullcontext
        with maybe_init_scope():
            with ops.name_scope(name, 'Variable', [] if init_from_fn else [initial_value], skip_on_eager=False) as name:
                handle_name = ops.name_from_scope_name(name)
                if self._in_graph_mode:
                    shared_name = handle_name
                    unique_id = shared_name
                else:
                    unique_id = '%s_%d' % (handle_name, ops.uid())
                    shared_name = None
                device_context_manager = ops.device if self._in_graph_mode else ops.NullContextmanager
                attr = attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(s=[compat.as_bytes('loc:@%s' % handle_name)]))
                with ops.get_default_graph()._attr_scope({'_class': attr}):
                    with ops.name_scope('Initializer'), device_context_manager(None):
                        if init_from_fn:
                            initial_value = initial_value()
                        if isinstance(initial_value, trackable.CheckpointInitialValue):
                            self._maybe_initialize_trackable()
                            self._update_uid = initial_value.checkpoint_position.restore_uid
                            initial_value = initial_value.wrapped_value
                        initial_value = ops.convert_to_tensor(initial_value, name='initial_value', dtype=dtype)
                    if shape is not None:
                        if not initial_value.shape.is_compatible_with(shape):
                            raise ValueError(f"In this `tf.Variable` creation, the initial value's shape ({initial_value.shape}) is not compatible with the explicitly supplied `shape` argument ({shape}).")
                    else:
                        shape = initial_value.shape
                    handle = eager_safe_variable_handle(initial_value=initial_value, shape=shape, shared_name=shared_name, name=name, graph_mode=self._in_graph_mode)
                    handle._parent_trackable = weakref.ref(self)
                    handle._name = handle_name + ':0'
                    handle._unique_id = unique_id
                if self._in_graph_mode and initial_value is not None and (initial_value.op._get_control_flow_context() is not None):
                    raise ValueError(f'The `initial_value` passed to `tf.Variable` {name} is from inside a control-flow  construct, such as a loop or conditional. When creating a `tf.Variable` inside a loop or conditional, use a lambda as the `initial_value`. Got: initial_value=({initial_value})')
                dtype = initial_value.dtype.base_dtype
                if self._in_graph_mode:
                    with ops.name_scope('IsInitialized'):
                        is_initialized_op = gen_resource_variable_ops.var_is_initialized_op(handle)
                    if initial_value is not None:
                        with ops.name_scope('Assign') as n, ops.colocate_with(None, ignore_existing=True), ops.device(handle.device):
                            initializer_op = gen_resource_variable_ops.assign_variable_op(handle, variables._try_guard_against_uninitialized_dependencies(name, initial_value), name=n)
                    with ops.name_scope('Read'):
                        with ops.device(handle.device):
                            value = gen_resource_variable_ops.read_variable_op(handle, dtype)
                            _maybe_set_handle_data(dtype, handle, value)
                        graph_element = value
                        if caching_device is not None:
                            with ops.colocate_with(None, ignore_existing=True):
                                with ops.device(caching_device):
                                    cached_value = array_ops.identity(value)
                        else:
                            cached_value = None
                else:
                    gen_resource_variable_ops.assign_variable_op(handle, initial_value)
                    is_initialized_op = None
                    initializer_op = None
                    graph_element = None
                    if caching_device:
                        with ops.device(caching_device):
                            cached_value = gen_resource_variable_ops.read_variable_op(handle, dtype)
                            _maybe_set_handle_data(dtype, handle, cached_value)
                    else:
                        cached_value = None
                if cached_value is not None:
                    cached_value._cached_variable = weakref.ref(self)
                if self._in_graph_mode:
                    ops.add_to_collections(collections, self)
                elif ops.GraphKeys.GLOBAL_STEP in collections:
                    ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, self)
            initial_value = initial_value if self._in_graph_mode else None
            super(ResourceVariable, self).__init__(trainable=trainable, shape=shape, dtype=dtype, handle=handle, synchronization=synchronization, constraint=constraint, aggregation=aggregation, distribute_strategy=distribute_strategy, name=name, unique_id=unique_id, handle_name=handle_name, graph_element=graph_element, initial_value=initial_value, initializer_op=initializer_op, is_initialized_op=is_initialized_op, cached_value=cached_value, caching_device=caching_device, validate_shape=validate_shape)

    def _init_from_proto(self, variable_def, import_scope=None, validate_shape=True):
        if False:
            return 10
        'Initializes from `VariableDef` proto.'
        assert not context.executing_eagerly()
        self._in_graph_mode = True
        assert isinstance(variable_def, variable_pb2.VariableDef)
        if not variable_def.is_resource:
            raise ValueError(f'The `variable_def` you passed to `tf.Variable` is Trying to restore a TF 1.x Reference Variable as a TF 2.x ResourceVariable. This is unsupported. Got variable_def={variable_def}')
        g = ops.get_default_graph()
        self._handle = g.as_graph_element(ops.prepend_name_scope(variable_def.variable_name, import_scope=import_scope), allow_operation=False)
        self._shape = tensor_shape.TensorShape(self._handle.op.get_attr('shape'))
        self._handle_name = self._handle.name
        self._unique_id = self._handle_name
        self._initializer_op = g.as_graph_element(ops.prepend_name_scope(variable_def.initializer_name, import_scope=import_scope))
        if hasattr(variable_def, 'initial_value_name') and variable_def.initial_value_name:
            self._initial_value = g.as_graph_element(ops.prepend_name_scope(variable_def.initial_value_name, import_scope=import_scope))
        else:
            self._initial_value = None
        (synchronization, aggregation, trainable) = variables.validate_synchronization_aggregation_trainable(variable_def.synchronization, variable_def.aggregation, variable_def.trainable, variable_def.variable_name)
        self._synchronization = synchronization
        self._aggregation = aggregation
        self._trainable = trainable
        if variable_def.snapshot_name:
            snapshot = g.as_graph_element(ops.prepend_name_scope(variable_def.snapshot_name, import_scope=import_scope))
            if snapshot.op.type != 'ReadVariableOp':
                self._cached_value = snapshot
            else:
                self._cached_value = None
            while snapshot.op.type != 'ReadVariableOp':
                snapshot = snapshot.op.inputs[0]
            self._graph_element = snapshot
        else:
            self._cached_value = None
            self._graph_element = g.get_tensor_by_name(self._handle.op.name + '/Read/ReadVariableOp:0')
        if variable_def.HasField('save_slice_info_def'):
            self._save_slice_info = variables.Variable.SaveSliceInfo(save_slice_info_def=variable_def.save_slice_info_def, import_scope=import_scope)
        else:
            self._save_slice_info = None
        self._caching_device = None
        self._dtype = dtypes.as_dtype(self._handle.op.get_attr('dtype'))
        self._constraint = None
        self._validate_shape = validate_shape

    def _init_from_handle(self, trainable=None, shape=None, dtype=None, handle=None):
        if False:
            print('Hello World!')
        handle_data = get_eager_safe_handle_data(handle)
        if not handle_data.is_set:
            handle_data = handle_data_util.create_handle_data(shape, dtype)
            handle_data_util.set_handle_data(handle, handle_data)
        if hasattr(handle, '_name') and isinstance(handle._name, str):
            handle_name = handle._name.rstrip(':0')
        else:
            handle_name = None
        unique_id = getattr(handle, '_unique_id', None)
        super().__init__(trainable=trainable, shape=shape, dtype=dtype, handle=handle, unique_id=unique_id, handle_name=handle_name)

class UninitializedVariable(BaseResourceVariable):
    """A variable with no initializer."""

    def __init__(self, trainable=None, caching_device=None, name=None, shape=None, dtype=None, constraint=None, synchronization=None, aggregation=None, extra_handle_data=None, distribute_strategy=None, **unused_kwargs):
        if False:
            while True:
                i = 10
        "Creates the variable handle.\n\n    Args:\n      trainable: If `True`, GradientTapes automatically watch uses of this\n        Variable.\n      caching_device: Optional device string or function describing where the\n        Variable should be cached for reading.  Defaults to the Variable's\n        device.  If not `None`, caches on another device.  Typical use is to\n        cache on the device where the Ops using the Variable reside, to\n        deduplicate copying through `Switch` and other conditional statements.\n      name: Optional name for the variable. Defaults to `'Variable'` and gets\n        uniquified automatically.\n      shape: The variable's shape.\n      dtype: The variable's dtype.\n      constraint: An optional projection function to be applied to the variable\n        after being updated by an `Optimizer` (e.g. used to implement norm\n        constraints or value constraints for layer weights). The function must\n        take as input the unprojected Tensor representing the value of the\n        variable and return the Tensor for the projected value (which must have\n        the same shape). Constraints are not safe to use when doing asynchronous\n        distributed training.\n      synchronization: Indicates when a distributed a variable will be\n        aggregated. Accepted values are constants defined in the class\n        `tf.VariableSynchronization`. By default the synchronization is set to\n        `AUTO` and the current `DistributionStrategy` chooses when to\n        synchronize.\n      aggregation: Indicates how a distributed variable will be aggregated.\n        Accepted values are constants defined in the class\n        `tf.VariableAggregation`.\n      extra_handle_data: Optional, another resource handle or Tensor with handle\n        data to merge with `shape` and `dtype`.\n      distribute_strategy: The tf.distribute.Strategy this variable is being\n        created inside of.\n    "
        with ops.init_scope():
            self._in_graph_mode = not context.executing_eagerly()
            with ops.name_scope(name, 'Variable', skip_on_eager=False) as name:
                handle_name = ops.name_from_scope_name(name)
                if self._in_graph_mode:
                    shared_name = handle_name
                    unique_id = shared_name
                else:
                    unique_id = '%s_%d' % (handle_name, ops.uid())
                    shared_name = None
                handle = _variable_handle_from_shape_and_dtype(shape=shape, dtype=dtype, shared_name=shared_name, name=name, graph_mode=self._in_graph_mode, initial_value=extra_handle_data)
                handle._parent_trackable = weakref.ref(self)
                handle._name = handle_name + ':0'
                handle._unique_id = unique_id
                if self._in_graph_mode:
                    with ops.name_scope('Read'):
                        with ops.device(handle.device):
                            value = gen_resource_variable_ops.read_variable_op(handle, dtype)
                            _maybe_set_handle_data(dtype, handle, value)
                        graph_element = value
                    ops.add_to_collection(ops.GraphKeys.GLOBAL_VARIABLES, self)
                else:
                    graph_element = None
        super(UninitializedVariable, self).__init__(distribute_strategy=distribute_strategy, shape=shape, dtype=dtype, unique_id=unique_id, handle_name=handle_name, constraint=constraint, handle=handle, graph_element=graph_element, trainable=trainable, synchronization=synchronization, aggregation=aggregation, in_graph_mode=self._in_graph_mode, **unused_kwargs)
_pywrap_utils.RegisterType('ResourceVariable', ResourceVariable)
math_ops._resource_variable_type = ResourceVariable

def _dense_var_to_tensor(var, dtype=None, name=None, as_ref=False):
    if False:
        print('Hello World!')
    return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)
tensor_conversion_registry.register_tensor_conversion_function(BaseResourceVariable, _dense_var_to_tensor)

class _UnreadVariable(BaseResourceVariable):
    """Represents a future for a read of a variable.

  Pretends to be the tensor if anyone looks.
  """

    def __init__(self, handle, dtype, shape, in_graph_mode, parent_op, unique_id):
        if False:
            i = 10
            return i + 15
        if isinstance(handle, ops.EagerTensor):
            handle_name = ''
        else:
            handle_name = handle.name
        if context.executing_eagerly() or ops.inside_function():
            graph_element = None
        else:
            with ops.control_dependencies([parent_op]):
                graph_element = gen_resource_variable_ops.read_variable_op(handle, dtype)
                _maybe_set_handle_data(dtype, handle, graph_element)
        super(_UnreadVariable, self).__init__(handle=handle, shape=shape, handle_name=handle_name, unique_id=unique_id, dtype=dtype, graph_element=graph_element)
        self._parent_op = parent_op

    @property
    def name(self):
        if False:
            return 10
        if self._in_graph_mode:
            return self._parent_op.name
        else:
            return 'UnreadVariable'

    def value(self):
        if False:
            while True:
                i = 10
        return self._read_variable_op()

    def read_value(self):
        if False:
            return 10
        return self._read_variable_op()

    def _read_variable_op(self):
        if False:
            while True:
                i = 10
        with ops.control_dependencies([self._parent_op]):
            result = gen_resource_variable_ops.read_variable_op(self._handle, self._dtype)
            _maybe_set_handle_data(self._dtype, self._handle, result)
            return result

    def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
        if False:
            print('Hello World!')
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).assign_sub(delta, use_locking, name, read_value)

    def assign_add(self, delta, use_locking=None, name=None, read_value=True):
        if False:
            print('Hello World!')
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).assign_add(delta, use_locking, name, read_value)

    def assign(self, value, use_locking=None, name=None, read_value=True):
        if False:
            while True:
                i = 10
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).assign(value, use_locking, name, read_value)

    def scatter_sub(self, sparse_delta, use_locking=False, name=None):
        if False:
            return 10
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_sub(sparse_delta, use_locking, name)

    def scatter_add(self, sparse_delta, use_locking=False, name=None):
        if False:
            while True:
                i = 10
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_add(sparse_delta, use_locking, name)

    def scatter_max(self, sparse_delta, use_locking=False, name=None):
        if False:
            print('Hello World!')
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_max(sparse_delta, use_locking, name)

    def scatter_min(self, sparse_delta, use_locking=False, name=None):
        if False:
            i = 10
            return i + 15
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_min(sparse_delta, use_locking, name)

    def scatter_mul(self, sparse_delta, use_locking=False, name=None):
        if False:
            print('Hello World!')
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_mul(sparse_delta, use_locking, name)

    def scatter_div(self, sparse_delta, use_locking=False, name=None):
        if False:
            while True:
                i = 10
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_div(sparse_delta, use_locking, name)

    def scatter_update(self, sparse_delta, use_locking=False, name=None):
        if False:
            for i in range(10):
                print('nop')
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_update(sparse_delta, use_locking, name)

    def batch_scatter_update(self, sparse_delta, use_locking=False, name=None):
        if False:
            for i in range(10):
                print('nop')
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).batch_scatter_update(sparse_delta, use_locking, name)

    def scatter_nd_sub(self, indices, updates, name=None):
        if False:
            while True:
                i = 10
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_nd_sub(indices, updates, name)

    def scatter_nd_add(self, indices, updates, name=None):
        if False:
            for i in range(10):
                print('nop')
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_nd_add(indices, updates, name)

    def scatter_nd_update(self, indices, updates, name=None):
        if False:
            return 10
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_nd_update(indices, updates, name)

    def scatter_nd_max(self, indices, updates, name=None):
        if False:
            print('Hello World!')
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_nd_max(indices, updates, name)

    def scatter_nd_min(self, indices, updates, name=None):
        if False:
            for i in range(10):
                print('nop')
        with ops.control_dependencies([self._parent_op]):
            return super(_UnreadVariable, self).scatter_nd_min(indices, updates, name)

    @property
    def op(self) -> ops.Operation:
        if False:
            print('Hello World!')
        'The op for this variable.'
        return self._parent_op

@ops.RegisterGradient('ReadVariableOp')
def _ReadGrad(_, grad):
    if False:
        i = 10
        return i + 15
    'Gradient for read op.'
    return grad

def variable_shape(handle, out_type=dtypes.int32):
    if False:
        return 10
    handle_data = get_eager_safe_handle_data(handle)
    if handle_data is None or not handle_data.is_set:
        return gen_resource_variable_ops.variable_shape(handle, out_type=out_type)
    shape_proto = handle_data.shape_and_type[0].shape
    if shape_proto.unknown_rank or any((x.size == -1 for x in shape_proto.dim)):
        return gen_resource_variable_ops.variable_shape(handle, out_type=out_type)
    return constant_op.constant([x.size for x in shape_proto.dim], dtype=out_type)

@ops.RegisterGradient('ResourceGather')
def _GatherGrad(op, grad):
    if False:
        while True:
            i = 10
    'Gradient for gather op.'
    handle = op.inputs[0]
    indices = op.inputs[1]
    params_shape = variable_shape(handle)
    size = array_ops.expand_dims(array_ops.size(indices), 0)
    values_shape = array_ops.concat([size, params_shape[1:]], 0)
    values = array_ops.reshape(grad, values_shape)
    indices = array_ops.reshape(indices, size)
    return (indexed_slices.IndexedSlices(values, indices, params_shape), None)

@tf_export('__internal__.ops.is_resource_variable', v1=[])
def is_resource_variable(var):
    if False:
        return 10
    '"Returns True if `var` is to be considered a ResourceVariable.'
    return isinstance(var, BaseResourceVariable) or hasattr(var, '_should_act_as_resource_variable')

def copy_to_graph_uninitialized(var):
    if False:
        for i in range(10):
            print('nop')
    'Copies an existing variable to a new graph, with no initializer.'
    new_variable = UninitializedVariable(trainable=var.trainable, constraint=var._constraint, shape=var.shape, dtype=var.dtype, name=var._shared_name, synchronization=var.synchronization, aggregation=var.aggregation, extra_handle_data=var.handle)
    new_variable._maybe_initialize_trackable()
    return new_variable
ops.NotDifferentiable('Assert')
ops.NotDifferentiable('VarIsInitializedOp')
ops.NotDifferentiable('VariableShape')

class StructurePattern:
    pass

class PLeaf(StructurePattern):
    """Represents a singleton leaf StructurePattern."""

    def __new__(cls):
        if False:
            print('Hello World!')
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
        return cls.instance

class PList(StructurePattern):
    """Represents a list of StructurePatterns."""

    def __init__(self, *components):
        if False:
            for i in range(10):
                print('nop')
        self.components = list(components)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return isinstance(other, PList) and self.components == other.components

class VariableSpec(tensor_module.DenseSpec):
    """Describes a tf.Variable.

  A `VariableSpec` provides metadata describing the `tf.Variable` objects
  accepted or returned by TensorFlow 2.x APIs.
  """
    __slots__ = ['trainable', 'alias_id']
    value_type = property(lambda self: ResourceVariable)

    def __init__(self, shape, dtype=dtypes.float32, trainable=True, alias_id=None):
        if False:
            return 10
        super(VariableSpec, self).__init__(shape, dtype=dtype)
        self.trainable = trainable
        self.alias_id = alias_id

    def is_compatible_with(self, spec_or_value):
        if False:
            for i in range(10):
                print('nop')
        'Returns True if `spec_or_value` is compatible with this `VariableSpec`.\n\n    `spec_or_value` is considered to be compatible with this `VariableSpec` if\n\n    * `spec_or_value` is a `Variable` or `VariableSpec`,\n    * their shapes are compatible,\n    * their dtypes are the same,\n    * they are both trainable or not trainable.\n    * they share the same alias_id if `spec_or_value` is a `VariableSpec`.\n\n    Example:\n\n    >>> v = tf.Variable([1., 2., 3.])\n    >>> spec = VariableSpec([None])\n    >>> spec.is_compatible_with(v)\n    True\n    >>> v = tf.Variable(1)\n    >>> spec.is_compatible_with(v)\n    False\n\n    Args:\n      spec_or_value: A VariableSpec or Variable to compare against.\n\n    Returns:\n      True if `spec_or_value` is compatible with this `VariableSpec`.\n    '
        if not isinstance(spec_or_value, (type(self), self.value_type)):
            return False
        compatible = self.shape.is_compatible_with(spec_or_value.shape) and self.dtype == spec_or_value.dtype and (self.trainable == spec_or_value.trainable)
        if isinstance(spec_or_value, type(self)):
            return compatible and self.alias_id == spec_or_value.alias_id
        return compatible

    @classmethod
    def from_value(cls, value):
        if False:
            i = 10
            return i + 15
        "Creates a `VariableSpec` from the given `Variable`.\n\n    `value`'s shape, dtype, and trainable attributes will be used to create\n    the new `VariableSpec`.\n\n    Example:\n\n    >>> v = tf.Variable([1., 2., 3.])\n    >>> VariableSpec.from_value(v)\n    VariableSpec(shape=(3,), dtype=tf.float32, trainable=True, alias_id=None)\n\n    Args:\n      value: A Variable.\n\n    Returns:\n      A `VariableSpec` created from `value`.\n    "
        return cls(value.shape, dtype=value.dtype, trainable=value.trainable)

    def _to_components(self, value):
        if False:
            while True:
                i = 10
        return [value.handle]

    def _from_components(self, components):
        if False:
            i = 10
            return i + 15
        if not isinstance(components, (list, tuple)):
            raise TypeError(f'Components of a ResourceVariable must be a list or tuple, got f{components} instead.')
        if len(components) != 1:
            raise ValueError(f'Components of a ResourceVariable must only contain its resource handle, got f{components} instead.')
        handle = components[0]
        if not isinstance(handle, tensor_module.Tensor) or handle.dtype != dtypes.resource:
            raise ValueError(f'The handle of a ResourceVariable must be a resource tensor, got {handle} instead.')
        return ResourceVariable(trainable=self.trainable, shape=self.shape, dtype=self.dtype, handle=handle)

    @property
    def _component_specs(self):
        if False:
            i = 10
            return i + 15
        return [tensor_module.TensorSpec([], dtypes.DType(dtypes.resource._type_enum, dtypes.HandleData(alias_id=self.alias_id)))]

    def _serialize(self):
        if False:
            print('Hello World!')
        return (self.shape, self.dtype, self.trainable, self.alias_id)

    def is_subtype_of(self, other):
        if False:
            while True:
                i = 10
        if type(self) is not type(other):
            return False
        if self.alias_id is None and other.alias_id is None:
            return super().is_subtype_of(other)
        if self.alias_id is None or other.alias_id is None:
            raise NotImplementedError(f"VariableSpec.is_subtype_of doesn't support alias_id=None, got self: {self} and other: {other}.")
        return super().is_subtype_of(other)

    def most_specific_common_supertype(self, others):
        if False:
            print('Hello World!')
        if any((type(self) is not type(other) for other in others)):
            return None
        if self.alias_id is None and all((other.alias_id is None for other in others)):
            return super().most_specific_common_supertype(others)
        if self.alias_id is None or any((other.alias_id is None for other in others)):
            raise NotImplementedError(f"VariableSpec.most_specific_common_supertype doesn't support alias_id=None, got self: {self} and others: {others}.")
        return super().most_specific_common_supertype(others)

    def placeholder_value(self, placeholder_context):
        if False:
            return 10
        if placeholder_context.unnest_only:
            return self
        name = self.name or placeholder_context.naming_scope
        context_graph = placeholder_context.context_graph
        if placeholder_context.has_placeholder(self.alias_id):
            variable = placeholder_context.get_placeholder(self.alias_id)
        else:
            spec = tensor_module.TensorSpec([], dtypes.resource)
            spec_context = trace_type.InternalPlaceholderContext(context_graph.outer_graph)
            spec_context.update_naming_scope(name)
            placeholder = spec.placeholder_value(spec_context)
            variable = self._from_components([placeholder])
            if self.alias_id is not None:
                placeholder_context.add_placeholder(self.alias_id, variable)
        placeholder = context_graph.capture(variable.handle, name=name)
        placeholder.op._set_attr('_user_specified_name', attr_value_pb2.AttrValue(s=compat.as_bytes(name)))
        return variable

    def to_tensors(self, value):
        if False:
            return 10
        assert isinstance(value, BaseResourceVariable)
        variable_accessed(value)
        return [value.handle]

    def cast(self, value, _):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(value, BaseResourceVariable)
        return value

    def _get_structure(self):
        if False:
            while True:
                i = 10
        return PList(PLeaf(), PLeaf(), PLeaf(), PLeaf())

    def __repr__(self):
        if False:
            return 10
        return f'{type(self).__name__}(shape={self.shape}, dtype={self.dtype!r}, trainable={self.trainable!r}, alias_id={self.alias_id!r})'

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash((self.shape, self.dtype, self.trainable, self.alias_id))

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return type(self) is type(other) and self.shape == other.shape and (self.dtype == other.dtype) and (self.trainable == other.trainable) and (self.alias_id == other.alias_id)
nested_structure_coder.register_codec(nested_structure_coder.BuiltInTypeSpecCodec(VariableSpec, struct_pb2.TypeSpecProto.VARIABLE_SPEC))
_pywrap_utils.RegisterType('VariableSpec', VariableSpec)

def write_object_proto_for_resource_variable(resource_variable, proto, options, enforce_naming=True):
    if False:
        print('Hello World!')
    "Writes additional information of the variable into the SavedObject proto.\n\n  This allows users to define a `hook` to provide extra information of the\n  variable to the SavedObject.\n\n  For example, DistributedVariable class would fill in components in the\n  distributed context.\n\n  Args:\n    resource_variable: A `ResourceVariable` or `DistributedValue` that has the\n      information to be saved into the proto.\n    proto: `SavedObject` proto to update.\n    options: A `SaveOption` instance that configures save behavior.\n    enforce_naming: A bool determining whether to check that names end in the\n      expected string ':0'\n  "
    proto.variable.SetInParent()
    if enforce_naming and (not resource_variable.name.endswith(':0')):
        raise ValueError(f"Cowardly refusing to save variable {resource_variable.name} because of unexpected suffix in the name (expected ':0')which won't be restored.")
    proto.variable.name = tensor_module.get_op_name(resource_variable.name)
    proto.variable.trainable = resource_variable.trainable
    proto.variable.dtype = resource_variable.dtype.as_datatype_enum
    proto.variable.synchronization = resource_variable.synchronization.value
    proto.variable.aggregation = resource_variable.aggregation.value
    proto.variable.shape.CopyFrom(resource_variable.shape.as_proto())
    if options.experimental_variable_policy._save_variable_devices():
        if hasattr(resource_variable, 'device'):
            proto.variable.device = resource_variable.device