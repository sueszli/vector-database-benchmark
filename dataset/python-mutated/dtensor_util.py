"""Utilities for strategies that are backed by DTensor."""
from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import input_util
from tensorflow.dtensor.python import layout
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2
DEFAULT_BATCH_MESH_DIM_NAME = 'batch'

class DTensorDistributedValue(values_lib.DistributedValues):
    """DistributedValue backed by a DTensor instance.

  This class is useful to align the interface between DTensor and tf.distribute.
  Most of the tf.distribute API will accept/return DistributedValue, whereas
  DTensor low level API will only accept DTensor instance. In order to avoid
  the conversion back and forth between DistributedValue and DTensor, we
  introduce this class so that it can work with both side.
  """

    def __init__(self, dtensor):
        if False:
            print('Hello World!')
        if context.executing_eagerly():
            if not d_api.is_dtensor(dtensor):
                raise ValueError(f'The DTensorDistributedValue can only be built with DTensor instance, got {type(dtensor)}')
            super().__init__(d_api.unpack(dtensor))
        else:
            super().__init__([dtensor])
        self._dtensor = dtensor

    def get_dtensor(self):
        if False:
            while True:
                i = 10
        return self._dtensor

    @property
    def values(self):
        if False:
            print('Hello World!')
        return self._values

def _dtensor_distributed_value_to_tensor(var, dtype=None, name=None, as_ref=False):
    if False:
        return 10
    del name
    dtensor = var.get_dtensor()
    if dtype is not None and (not dtype.is_compatible_with(dtensor.dtype)):
        raise ValueError('Incompatible type conversion requested to type {!r} for variable of type {!r}'.format(dtype.name, dtensor.dtype.name))
    if as_ref:
        raise NotImplementedError("PerReplica doesn't support being used as a reference.")
    return dtensor
tensor_conversion_registry.register_tensor_conversion_function(DTensorDistributedValue, _dtensor_distributed_value_to_tensor)

class DTensorReplicaContext(distribute_lib.ReplicaContext):
    """ReplicaContext for strategy that is backed by DTensor.

  Since the DTensor is operated in the global context, most of the methods from
  existing strategy ReplicaContext is not applicable since they need to access
  local values. For now most of the methods in this class will raise explicit
  error to user, and we will add more support for local values in future.
  """
    _UNSUPPORTED_ERROR_MSG = "Strategy that is backed by DTensor is run with a global context, and doesn't support operations for local context, like any call to merge/gather/reduce or local replica ID. Please use any strategy that is not backed by DTensor"

    def __init__(self, strategy):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(strategy, replica_id_in_sync_group=None)

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        distribute_lib._push_per_thread_mode(self._thread_context)
        summary_state = summary_ops_v2._summary_state
        self._summary_recording_distribution_strategy = summary_state.is_recording_distribution_strategy
        summary_state.is_recording_distribution_strategy = True

    @property
    def replica_id_in_sync_group(self):
        if False:
            i = 10
            return i + 15
        return 0

    @property
    def _replica_id(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

    def merge_call(self, merge_fn, args=(), kwargs=None):
        if False:
            return 10
        raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

    def all_reduce(self, reduce_op, value, options=None):
        if False:
            return 10
        raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

    def all_gather(self, value, axis, options=None):
        if False:
            while True:
                i = 10
        raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

    def _update(self, var, fn, args=(), kwargs=None, group=True):
        if False:
            while True:
                i = 10
        raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

def initialize_accelerator_system_once(device_type):
    if False:
        return 10
    if not accelerator_util.is_initialized():
        accelerator_util.initialize_accelerator_system(device_type, experimental_reset_context=True)

def convert_inputs_to_dtensor(inputs, mesh):
    if False:
        print('Hello World!')
    'Convert any input types to DTensor instance.'
    if isinstance(inputs, DTensorDistributedValue):
        return inputs.get_dtensor()
    elif isinstance(inputs, values_lib.DistributedValues):
        return convert_per_replica_to_dtensor(inputs, mesh)
    elif isinstance(inputs, input_util._DTensorIterator):
        return inputs
    elif tensor_util.is_tensor(inputs):
        if context.executing_eagerly():
            if d_api.is_dtensor(inputs):
                return inputs
            else:
                _raise_unsupported_input_type_error(inputs)
        else:
            return inputs
    else:
        _raise_unsupported_input_type_error(inputs)

def _raise_unsupported_input_type_error(inputs):
    if False:
        for i in range(10):
            print('nop')
    raise ValueError(f'Unsupported input types for MirroredStrategy. Please use `strategy.distribute_dataset` or `strategy.distribute_values_from_function` to distribute inputs. Received input type: {type(inputs)}')

def is_distributed_value(value):
    if False:
        print('Hello World!')
    return isinstance(value, values_lib.DistributedValues) or d_api.is_dtensor(value)

def convert_per_replica_to_dtensor(per_replica_value, mesh):
    if False:
        print('Hello World!')
    'Convert a PerReplica result to a DTensor instance.\n\n  Args:\n    per_replica_value: A PerReplica instance whose value will be converted\n      to DTensor.\n    mesh: The mesh used for layout creation.\n\n  Returns:\n    A DTensor instance that packed from per_replica_value with batch sharded\n      layout.\n  '
    values = per_replica_value.values
    if isinstance(values[0], (float, int)):
        rank = 0
    else:
        rank = len(values[0].shape)
    if rank == 0:
        result = []
        for v in values:
            result.append(array_ops.expand_dims_v2(v, axis=0))
        rank += 1
    else:
        result = list(values)
    batch_layout = layout.Layout.batch_sharded(mesh, batch_dim=DEFAULT_BATCH_MESH_DIM_NAME, rank=rank)
    return d_api.pack(result, batch_layout)

def dtensor_reduce(strategy, reduce_op, value, axis):
    if False:
        print('Hello World!')
    'Implement dtensor based strategy.reduce().'
    distribute_lib._require_cross_replica_or_default_context_extended(strategy.extended)
    if isinstance(reduce_op, str):
        reduce_op = reduce_util.ReduceOp(reduce_op.upper())
    distributed_input = is_distributed_value(value)
    if not distributed_input and axis is None:
        destinations = device_util.current() or strategy.extended._default_device or '/device:CPU:0'
        devices = cross_device_ops_lib.get_devices_from(destinations)
        with ops.device(devices[0]):
            return array_ops.identity(cross_device_ops_lib.reduce_non_distributed_value(reduce_op, value, destinations, strategy.num_replicas_in_sync))
    value = convert_inputs_to_dtensor(value, strategy._mesh)
    if reduce_op == reduce_util.ReduceOp.MEAN:
        reduce_op = math_ops.reduce_mean
    else:
        reduce_op = math_ops.reduce_sum
    if d_api.fetch_layout(value).is_fully_replicated():
        if axis is not None:
            value = reduce_op(value, axis=axis)
    else:
        new_shape = [strategy.num_replicas_in_sync, -1]
        if len(value.shape) > 1:
            new_shape.extend(array_ops.shape(value)[1:])
        value = array_ops.reshape(value, new_shape)
        if axis is not None:
            value = reduce_op(value, axis=axis + 1)
        value = reduce_op(value, axis=0)
    return value