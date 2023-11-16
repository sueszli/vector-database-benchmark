"""A Variable class that is replicated to logical cores for model parallelism."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import abc
import contextlib
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.distribute import tpu_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_tpu_partition_ops as tpu_partition_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable

def _on_device_update(update_fn, var, value, **kwargs):
    if False:
        return 10
    with ops.device(var.device):
        return update_fn(var, value, **kwargs)

class TPUReplicatedVariable(variables_lib.Variable):
    """Container for replicated `Variables` that are treated as a single variable.

  This class maintains a list of replicated variables that are stored on
  separate logic TPU devices. TF2XLA bridge accesses these variables as
  if they were a single variable.
  """

    def __init__(self, variables, name='TPUReplicatedVariable'):
        if False:
            print('Hello World!')
        'Treats `variables` as a replicated list of `tf.Variable`s.\n\n    Example:\n\n    ```\n    variables = [\n      tf.Variable(..., shape=(10, 100), dtype=tf.float32),\n      tf.Variable(..., shape=(10, 100), dtype=tf.float32),\n      tf.Variable(..., shape=(10, 100), dtype=tf.float32),\n      tf.Variable(..., shape=(10, 100), dtype=tf.float32),\n    ]\n    replicated_variable = TPUReplicatedVariable(variables)\n    assert replicated_variable.shape.as_list() == [10, 100]\n    ```\n\n    Args:\n      variables: A list of `ResourceVariable`s that comprise this replicated\n        variable. Variables should not be shared between different\n        `TPUReplicatedVariable` objects.\n      name: String. Name of this container. Defaults to "TPUReplicatedVariable".\n    '
        if not isinstance(variables, abc.Sequence) or not variables or any((not isinstance(v, variables_lib.Variable) for v in variables)):
            raise TypeError(f'Argument `variables` should be a non-empty list of `variables.Variable`s. Received {variables}')
        if any((v.dtype != variables[0].dtype for v in variables)):
            raise ValueError(f'All elements in argument `variables` must have the same dtype. Received dtypes: {[v.dtype for v in variables]}')
        if any((v.shape != variables[0].shape for v in variables)):
            raise ValueError(f'All elements in argument `variables` must have the same shape. Received shapes: {[v.shape for v in variables]}')
        self._vars = variables
        self._name = name
        self._common_name = self._name.split(':')[0]
        self._cached_value = None

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        'Return an iterable for accessing the underlying sharded variables.'
        return iter(self._vars)

    @property
    def name(self):
        if False:
            return 10
        'The name of this object. Used for checkpointing.'
        return self._name

    @property
    def dtype(self):
        if False:
            print('Hello World!')
        'The dtype of all `Variable`s in this object.'
        return self._vars[0].dtype

    @property
    def is_initialized(self):
        if False:
            return 10
        return self._vars[0].is_initialized

    @property
    def trainable(self):
        if False:
            while True:
                i = 10
        return self._vars[0].trainable

    @property
    def device(self):
        if False:
            while True:
                i = 10
        'The device this variable is on.'
        return self._vars[0].device

    @contextlib.contextmanager
    def _handle_graph(self):
        if False:
            for i in range(10):
                print('nop')
        with self.handle.graph.as_default():
            yield

    @contextlib.contextmanager
    def _assign_dependencies(self):
        if False:
            print('Hello World!')
        if self._cached_value is not None:
            with ops.control_dependencies([self._cached_value]):
                yield
        else:
            yield

    @property
    def constraint(self):
        if False:
            i = 10
            return i + 15
        return self._vars[0].constraint

    @property
    def _in_graph_mode(self):
        if False:
            return 10
        return self._vars[0]._in_graph_mode

    @property
    def _unique_id(self):
        if False:
            for i in range(10):
                print('nop')
        return self._vars[0]._unique_id

    @property
    def graph(self):
        if False:
            print('Hello World!')
        return self._vars[0].graph

    @property
    def _shared_name(self):
        if False:
            i = 10
            return i + 15
        return self._common_name

    @property
    def synchronization(self):
        if False:
            while True:
                i = 10
        return variable_scope.VariableSynchronization.NONE

    @property
    def aggregation(self):
        if False:
            while True:
                i = 10
        return variable_scope.VariableAggregation.NONE

    @property
    def variables(self):
        if False:
            i = 10
            return i + 15
        'The list of `Variables`.'
        if save_context.in_save_context():
            return [self._vars[0]]
        return self._vars

    def _export_to_saved_model_graph(self, object_map, tensor_map, options, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'For implementing `Trackable`.'
        first_var = self._vars[0]
        resource_list = first_var._export_to_saved_model_graph(object_map, tensor_map, options, **kwargs)
        for v in self._vars[1:]:
            object_map[v] = object_map[first_var]
            tensor_map[v.handle] = tensor_map[first_var.handle]
            resource_list.append(v.handle)
        object_map[self] = object_map[first_var]
        tensor_map[self] = tensor_map[first_var.handle]
        resource_list.append(self)
        return resource_list

    def _serialize_to_tensors(self):
        if False:
            i = 10
            return i + 15
        return {trackable.VARIABLE_VALUE_KEY: self._vars[0]}

    def _restore_from_tensors(self, restored_tensors):
        if False:
            return 10
        restored_tensor = restored_tensors[trackable.VARIABLE_VALUE_KEY]
        return self.assign(restored_tensor)

    def _copy_trackable_to_cpu(self, object_map):
        if False:
            print('Hello World!')
        'For implementing `Trackable`.'
        if self in object_map:
            for v in self._vars:
                v._copy_trackable_to_cpu(object_map)
        else:
            copied_vars = []
            for v in self._vars:
                v._copy_trackable_to_cpu(object_map)
                copied_vars.append(object_map[v])
            new_var = TPUReplicatedVariable(copied_vars, name=self.name)
            object_map[self] = new_var

    @property
    def shape(self):
        if False:
            while True:
                i = 10
        return self._vars[0].shape

    @property
    def handle(self):
        if False:
            return 10
        if save_context.in_save_context() or context.executing_eagerly():
            return self._vars[0].handle
        if tpu_util.enclosing_tpu_context() is None:
            raise NotImplementedError('TPUReplicatedVariable.handle is not available outside tpu context or save context')
        else:
            with tpu_util.outside_or_skip_tpu_context():
                packed_var = getattr(self, '_packed_var', None)
                if packed_var is None or config.get_soft_device_placement():
                    tensor = tpu_partition_ops.tpu_partitioned_input_v2([v.handle for v in self._vars], partition_dims=[], is_packed=False)
                else:
                    tensor = tpu_partition_ops.tpu_partitioned_input_v2([packed_var.packed_handle], partition_dims=[], is_packed=True)
            return xla_sharding.replicate(tensor)

    def _read_variable_op(self):
        if False:
            return 10
        return gen_resource_variable_ops.read_variable_op(self.handle, self.dtype)

    def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
        if False:
            i = 10
            return i + 15
        'Converts a variable to a tensor.'
        if tpu_util.enclosing_tpu_context() is None:
            return self.read_value()
        else:
            return self._read_variable_op()

    def read_value(self):
        if False:
            print('Hello World!')
        return self._vars[0].read_value()

    def _update(self, update_fn, value, **kwargs):
        if False:
            print('Hello World!')
        'Converts the value to tensor and updates the variable list.'
        input_tensor = ops.convert_to_tensor(value, name='value_in_tensor', dtype=self.dtype)
        return control_flow_ops.group(*tuple((_on_device_update(update_fn, v, input_tensor, **kwargs) for v in self.variables)))

    def assign(self, value, use_locking=False, name=None, read_value=True):
        if False:
            return 10
        if tpu_util.enclosing_tpu_context() is None or context.executing_eagerly():
            assign_fn = lambda var, *a, **ka: var.assign(*a, **ka)
            return self._update(assign_fn, value=value, use_locking=use_locking, name=name, read_value=read_value)
        else:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_variable_op)(self, value=value, use_locking=use_locking, name=name, read_value=read_value)

    def assign_sub(self, value, use_locking=False, name=None, read_value=True):
        if False:
            i = 10
            return i + 15
        if tpu_util.enclosing_tpu_context() is None or context.executing_eagerly():
            assign_sub_fn = lambda var, *a, **ka: var.assign_sub(*a, **ka)
            return self._update(assign_sub_fn, value=value, use_locking=use_locking, name=name, read_value=read_value)
        else:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_sub_variable_op)(self, value=value, use_locking=use_locking, name=name, read_value=read_value)

    def assign_add(self, value, use_locking=False, name=None, read_value=True):
        if False:
            i = 10
            return i + 15
        if tpu_util.enclosing_tpu_context() is None or context.executing_eagerly():
            assign_add_fn = lambda var, *a, **ka: var.assign_add(*a, **ka)
            return self._update(assign_add_fn, value=value, use_locking=use_locking, name=name, read_value=read_value)
        else:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_add_variable_op)(self, value=value, use_locking=use_locking, name=name, read_value=read_value)

    def __str__(self):
        if False:
            print('Hello World!')
        debug_str = ',\n'.join(('  %d: %s' % (i, v) for (i, v) in enumerate(self._vars)))
        return '%s:{\n%s\n}' % (self.__class__.__name__, debug_str)

    def __repr__(self):
        if False:
            return 10
        debug_repr = ',\n'.join(('  %d: %r' % (i, v) for (i, v) in enumerate(self._vars)))
        return '%s:{\n%s\n}' % (self.__class__.__name__, debug_repr)

def _tensor_conversion_tpu_replicated_var(var, dtype=None, name=None, as_ref=False):
    if False:
        while True:
            i = 10
    return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)
tensor_conversion_registry.register_tensor_conversion_function(TPUReplicatedVariable, _tensor_conversion_tpu_replicated_var)