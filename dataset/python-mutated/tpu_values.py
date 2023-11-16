"""Various classes representing TPU distributed values.

Note that the tests are in values_test.py .

"""
from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.distribute import tpu_replicated_variable
from tensorflow.python.distribute import tpu_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
_scatter_error_msg = '{op_name} is only supported for distributed variable (variable created within certain `tf.distribute.Strategy` scope) with NONE  aggregation, got: {aggregation}.'

class TPUVariableMixin(object):
    """Mixin for TPU variables."""

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(TPUVariableMixin, self).__init__(*args, **kwargs)
        if ops.executing_eagerly_outside_functions():
            self._handle_id = self._common_name + '_' + str(id(self._primary))
        else:
            self._handle_id = self._common_name

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        if tpu_util.enclosing_tpu_context() is None:
            return super(TPUVariableMixin, self).__getattr__(name)
        else:
            raise AttributeError(f'`TPUVariableMixin.{name}` not accessible within a TPU context.')

    def get(self):
        if False:
            print('Hello World!')
        if tpu_util.enclosing_tpu_context() is None:
            return super(TPUVariableMixin, self).get()
        else:
            raise NotImplementedError('`TPUVariableMixin.get()` is not supported within a TPU context.')

    def _get_as_operand(self):
        if False:
            print('Hello World!')
        return self.read_value()

    @property
    def handle(self):
        if False:
            return 10
        'The handle by which this variable can be accessed.'
        tpu_context = tpu_util.enclosing_tpu_context()
        if tpu_context is None or context.executing_eagerly():
            var = self._get_on_device_or_primary()
            if isinstance(var, packed.PackedVarAndDevice):
                return var.on_device_handle()
            else:
                return var.handle
        else:
            is_packed = self._packed_var is not None
            val = self._values
            if is_packed:
                val = [self._packed_var]
            return tpu_context.get_replicated_var_handle(self._common_name, self._handle_id, val, self._is_mirrored(), is_packed)

    @property
    def device(self):
        if False:
            print('Hello World!')
        return self.handle.device

    def _read_variable_op(self):
        if False:
            i = 10
            return i + 15
        'Reads the value of this variable.'
        if self.trainable:
            tape.variable_accessed(self)
        handle = self.handle
        if getattr(handle, 'is_packed', False):
            with ops.device(self._get_on_device_or_primary().device):
                return gen_resource_variable_ops.read_variable_op(handle, self.dtype)
        else:
            return gen_resource_variable_ops.read_variable_op(handle, self.dtype)

    def read_value(self):
        if False:
            for i in range(10):
                print('nop')
        if tpu_util.enclosing_tpu_context() is None:
            return super(TPUVariableMixin, self).read_value()
        else:
            return self._read_variable_op()

    def value(self):
        if False:
            for i in range(10):
                print('nop')
        if tpu_util.enclosing_tpu_context() is None:
            return super(TPUVariableMixin, self).value()
        else:
            return self._read_variable_op()

    def _as_graph_element(self):
        if False:
            for i in range(10):
                print('nop')
        if tpu_util.enclosing_tpu_context() is None:
            return super(TPUVariableMixin, self)._as_graph_element()
        else:
            return None

    @property
    def op(self):
        if False:
            while True:
                i = 10
        if values_util.is_saving_non_distributed():
            return self._primary.op
        return values.DistributedVarOp(self._primary.op.name, self._primary.op.graph, self._primary.op.traceback, self._primary.op.type)

    def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
        if False:
            i = 10
            return i + 15
        'Converts a variable to a tensor.'
        if tpu_util.enclosing_tpu_context() is None:
            return super(TPUVariableMixin, self)._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)
        elif dtype is not None and dtype != self.dtype:
            return math_ops.cast(self.read_value(), dtype)
        else:
            return self.handle if as_ref else self.read_value()

class TPUDistributedVariable(TPUVariableMixin, values.DistributedVariable):
    """DistributedVariable subclass for TPUStrategy."""

    def assign_sub(self, value, use_locking=False, name=None, read_value=True):
        if False:
            while True:
                i = 10
        if values_util.is_saving_non_distributed():
            return self._primary.assign_sub(value, use_locking, name, read_value)
        return self._policy.assign_sub(self, value, use_locking=use_locking, name=name, read_value=read_value)

    def assign_add(self, value, use_locking=False, name=None, read_value=True):
        if False:
            while True:
                i = 10
        if values_util.is_saving_non_distributed():
            return self._primary.assign_add(value, use_locking, name, read_value)
        return self._policy.assign_add(self, value, use_locking=use_locking, name=name, read_value=read_value)

    def assign(self, value, use_locking=False, name=None, read_value=True):
        if False:
            i = 10
            return i + 15
        if values_util.is_saving_non_distributed():
            return self._primary.assign(value, use_locking, name, read_value)
        return self._policy.assign(self, value, use_locking=use_locking, name=name, read_value=read_value)

    def scatter_sub(self, sparse_delta, use_locking=False, name=None):
        if False:
            return 10
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_sub(sparse_delta, use_locking, name)
        return self._policy.scatter_sub(self, sparse_delta, use_locking=use_locking, name=name)

    def scatter_add(self, sparse_delta, use_locking=False, name=None):
        if False:
            i = 10
            return i + 15
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_add(sparse_delta, use_locking, name)
        return self._policy.scatter_add(self, sparse_delta, use_locking=use_locking, name=name)

    def scatter_mul(self, sparse_delta, use_locking=False, name=None):
        if False:
            i = 10
            return i + 15
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_mul(sparse_delta, use_locking, name)
        return self._policy.scatter_mul(self, sparse_delta, use_locking=use_locking, name=name)

    def scatter_div(self, sparse_delta, use_locking=False, name=None):
        if False:
            for i in range(10):
                print('nop')
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_div(sparse_delta, use_locking, name)
        return self._policy.scatter_div(self, sparse_delta, use_locking=use_locking, name=name)

    def scatter_min(self, sparse_delta, use_locking=False, name=None):
        if False:
            i = 10
            return i + 15
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_min(sparse_delta, use_locking, name)
        return self._policy.scatter_min(self, sparse_delta, use_locking=use_locking, name=name)

    def scatter_max(self, sparse_delta, use_locking=False, name=None):
        if False:
            i = 10
            return i + 15
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_max(sparse_delta, use_locking, name)
        return self._policy.scatter_max(self, sparse_delta, use_locking=use_locking, name=name)

    def scatter_update(self, sparse_delta, use_locking=False, name=None):
        if False:
            while True:
                i = 10
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_update(sparse_delta, use_locking, name)
        return self._policy.scatter_update(self, sparse_delta, use_locking=use_locking, name=name)

class TPUMirroredVariable(TPUVariableMixin, values.MirroredVariable):
    """Holds a map from replica to TPU variables whose values are kept in sync."""

    def _is_replicated_or_sharded_to_logical_cores(self):
        if False:
            return 10
        'Returns whether each of the underlying variables is replicated or sharded to logical cores.\n\n    If True, the handles of the underlying variables are not available outside a\n    TPU context.\n    '
        return isinstance(self._primary, tpu_replicated_variable.TPUReplicatedVariable)

    @property
    def device(self):
        if False:
            print('Hello World!')
        if self._is_replicated_or_sharded_to_logical_cores() and tpu_util.enclosing_tpu_context() is None:
            return self._primary.device
        return super(TPUMirroredVariable, self).device

    def assign_sub(self, value, use_locking=False, name=None, read_value=True):
        if False:
            return 10
        tpu_context = tpu_util.enclosing_tpu_context()
        if self._is_replicated_or_sharded_to_logical_cores() and tpu_context is None:
            assign_sub_fn = lambda v, *a, **ka: v.assign_sub(*a, **ka)
            return self._update(update_fn=assign_sub_fn, value=value, use_locking=use_locking, name=name, read_value=read_value)
        if tpu_context and self.aggregation == variable_scope.VariableAggregation.NONE:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_sub_variable_op)(self, value=value, use_locking=use_locking, name=name, read_value=read_value)
        return assign_sub(self, value, use_locking=use_locking, name=name, read_value=read_value)

    def assign_add(self, value, use_locking=False, name=None, read_value=True):
        if False:
            return 10
        tpu_context = tpu_util.enclosing_tpu_context()
        if self._is_replicated_or_sharded_to_logical_cores() and tpu_context is None:
            assign_add_fn = lambda v, *a, **ka: v.assign_add(*a, **ka)
            return self._update(update_fn=assign_add_fn, value=value, use_locking=use_locking, name=name, read_value=read_value)
        if tpu_context and self.aggregation == variable_scope.VariableAggregation.NONE:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_add_variable_op)(self, value=value, use_locking=use_locking, name=name, read_value=read_value)
        return assign_add(self, value, use_locking=use_locking, name=name, read_value=read_value)

    def assign(self, value, use_locking=False, name=None, read_value=True):
        if False:
            for i in range(10):
                print('nop')
        tpu_context = tpu_util.enclosing_tpu_context()
        if self._is_replicated_or_sharded_to_logical_cores() and tpu_context is None:
            assign_fn = lambda v, *a, **ka: v.assign(*a, **ka)
            return self._update(update_fn=assign_fn, value=value, use_locking=use_locking, name=name, read_value=read_value)
        if tpu_util.enclosing_tpu_context() and self.aggregation == variable_scope.VariableAggregation.NONE:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_variable_op)(self, value=value, use_locking=use_locking, name=name, read_value=read_value)
        return assign(self, value, use_locking=use_locking, name=name, read_value=read_value)

    def scatter_sub(self, *args, **kwargs):
        if False:
            print('Hello World!')
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_sub(*args, **kwargs)
        raise NotImplementedError

    def scatter_add(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_add(*args, **kwargs)
        raise NotImplementedError

    def scatter_max(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_max(*args, **kwargs)
        raise NotImplementedError

    def scatter_min(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_min(*args, **kwargs)
        raise NotImplementedError

    def scatter_mul(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_mul(*args, **kwargs)
        raise NotImplementedError

    def scatter_div(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_div(*args, **kwargs)
        raise NotImplementedError

    def scatter_update(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_update(*args, **kwargs)
        raise NotImplementedError

class TPULazyDistributedVariable(TPUDistributedVariable):
    """TPU Mirrored variable to be initialized lazily in a batch."""

    def _initialize_if_uninitialized(self):
        if False:
            while True:
                i = 10
        if getattr(self, '_is_lazily_initialized', False):
            return
        self._lazy_scope.initialize_all()
        self._is_lazily_initialized = True

    def assign_sub(self, value, use_locking=False, name=None, read_value=True):
        if False:
            for i in range(10):
                print('nop')
        self._initialize_if_uninitialized()
        return super().assign_sub(value, use_locking, name, read_value)

    def assign_add(self, value, use_locking=False, name=None, read_value=True):
        if False:
            for i in range(10):
                print('nop')
        self._initialize_if_uninitialized()
        return super().assign_add(value, use_locking, name, read_value)

    def assign(self, value, use_locking=False, name=None, read_value=True):
        if False:
            i = 10
            return i + 15
        self._initialize_if_uninitialized()
        return super().assign(value, use_locking, name, read_value)

    def read_value(self):
        if False:
            return 10
        self._initialize_if_uninitialized()
        return super().read_value()

class TPUSyncOnReadVariable(TPUVariableMixin, values.SyncOnReadVariable):
    """Holds a map from replica to variables whose values are reduced on save."""

    def assign_sub(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if tpu_util.enclosing_tpu_context() is None:
            return values.SyncOnReadVariable.assign_sub(self, *args, **kwargs)
        else:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_sub_variable_op)(self, *args, **kwargs)

    def assign_add(self, *args, **kwargs):
        if False:
            return 10
        if tpu_util.enclosing_tpu_context() is None:
            return values.SyncOnReadVariable.assign_add(self, *args, **kwargs)
        else:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_add_variable_op)(self, *args, **kwargs)

    def assign(self, *args, **kwargs):
        if False:
            return 10
        if tpu_util.enclosing_tpu_context() is None:
            return values.SyncOnReadVariable.assign(self, *args, **kwargs)
        else:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_variable_op)(self, *args, **kwargs)

def assign_sub(var, value, use_locking=False, name=None, read_value=True):
    if False:
        while True:
            i = 10
    assign_sub_fn = tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_sub_variable_op)
    return var._update(update_fn=assign_sub_fn, value=value, use_locking=use_locking, name=name, read_value=read_value)

def assign_add(var, value, use_locking=False, name=None, read_value=True):
    if False:
        i = 10
        return i + 15
    assign_add_fn = tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_add_variable_op)
    return var._update(update_fn=assign_add_fn, value=value, use_locking=use_locking, name=name, read_value=read_value)

def assign(var, value, use_locking=False, name=None, read_value=True):
    if False:
        i = 10
        return i + 15
    assign_fn = tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_variable_op)
    return var._update(update_fn=assign_fn, value=value, use_locking=use_locking, name=name, read_value=read_value)

class TPUOnWritePolicy(values.OnWritePolicy):
    """Policy defined for `tf.VariableSynchronization.ON_WRITE` synchronization.

  This policy is created when `synchronization` is set to
  `tf.VariableSynchronization.AUTO` or `tf.VariableSynchronization.ON_WRITE`.
  """

    def assign_sub(self, var, value, use_locking=False, name=None, read_value=True):
        if False:
            print('Hello World!')
        if tpu_util.enclosing_tpu_context() and var.aggregation == variable_scope.VariableAggregation.NONE:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_sub_variable_op)(var, value=value, use_locking=use_locking, name=name, read_value=read_value)
        return assign_sub(var, value, use_locking=use_locking, name=name, read_value=read_value)

    def assign_add(self, var, value, use_locking=False, name=None, read_value=True):
        if False:
            print('Hello World!')
        if tpu_util.enclosing_tpu_context() and var.aggregation == variable_scope.VariableAggregation.NONE:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_add_variable_op)(var, value=value, use_locking=use_locking, name=name, read_value=read_value)
        return assign_add(var, value, use_locking=use_locking, name=name, read_value=read_value)

    def assign(self, var, value, use_locking=False, name=None, read_value=True):
        if False:
            for i in range(10):
                print('nop')
        if tpu_util.enclosing_tpu_context() and var.aggregation == variable_scope.VariableAggregation.NONE:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_variable_op)(var, value=value, use_locking=use_locking, name=name, read_value=read_value)
        return assign(var, value, use_locking=use_locking, name=name, read_value=read_value)

    def _scatter_xxx(self, raw_scater_xxx_fn, op_name, var, sparse_delta, use_locking=False, name=None):
        if False:
            print('Hello World!')
        scater_xxx_fn = tpu_util.make_raw_scatter_xxx_fn(raw_scater_xxx_fn)
        if tpu_util.enclosing_tpu_context():
            if self._aggregation != variable_scope.VariableAggregation.NONE:
                raise NotImplementedError(_scatter_error_msg.format(op_name=op_name, aggregation=self._aggregation))
            return scater_xxx_fn(var, sparse_delta=sparse_delta, use_locking=use_locking, name=name)
        else:
            return var._update(update_fn=scater_xxx_fn, value=sparse_delta, use_locking=use_locking, name=name)

    def scatter_sub(self, var, sparse_delta, use_locking=False, name=None):
        if False:
            print('Hello World!')
        return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_sub, 'scatter_sub', var, sparse_delta, use_locking, name)

    def scatter_add(self, var, sparse_delta, use_locking=False, name=None):
        if False:
            i = 10
            return i + 15
        return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_add, 'scatter_add', var, sparse_delta, use_locking, name)

    def scatter_max(self, var, sparse_delta, use_locking=False, name=None):
        if False:
            for i in range(10):
                print('nop')
        return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_max, 'scatter_max', var, sparse_delta, use_locking, name)

    def scatter_min(self, var, sparse_delta, use_locking=False, name=None):
        if False:
            for i in range(10):
                print('nop')
        return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_min, 'scatter_min', var, sparse_delta, use_locking, name)

    def scatter_mul(self, var, sparse_delta, use_locking=False, name=None):
        if False:
            return 10
        return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_mul, 'scatter_mul', var, sparse_delta, use_locking, name)

    def scatter_div(self, var, sparse_delta, use_locking=False, name=None):
        if False:
            print('Hello World!')
        return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_div, 'scatter_div', var, sparse_delta, use_locking, name)

    def scatter_update(self, var, sparse_delta, use_locking=False, name=None):
        if False:
            return 10
        return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_update, 'scatter_update', var, sparse_delta, use_locking, name)

class TPUOnReadPolicy(values.OnReadPolicy):
    """Policy defined for `tf.VariableSynchronization.ON_READ` synchronization.

  This policy is created when `synchronization` is set to
  `tf.VariableSynchronization.ON_READ` and `aggregation` is set to any of the
  values allowed by the `tf.VariableAggregation` enum such as `NONE`, `SUM`,
  `MEAN` or `ONLY_FIRST_REPLICA`when creating a `tf.Variable` in `tf.distribute`
  scope.
  """

    def assign_sub(self, var, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if tpu_util.enclosing_tpu_context() is None:
            return super(TPUOnReadPolicy, self).assign_sub(var, *args, **kwargs)
        else:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_sub_variable_op)(var, *args, **kwargs)

    def assign_add(self, var, *args, **kwargs):
        if False:
            return 10
        if tpu_util.enclosing_tpu_context() is None:
            return super(TPUOnReadPolicy, self).assign_add(var, *args, **kwargs)
        else:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_add_variable_op)(var, *args, **kwargs)

    def assign(self, var, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if tpu_util.enclosing_tpu_context() is None:
            return super(TPUOnReadPolicy, self).assign(var, *args, **kwargs)
        else:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_variable_op)(var, *args, **kwargs)

    def scatter_sub(self, *args, **kwargs):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def scatter_add(self, *args, **kwargs):
        if False:
            return 10
        raise NotImplementedError

    def scatter_max(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def scatter_min(self, *args, **kwargs):
        if False:
            return 10
        raise NotImplementedError

    def scatter_mul(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def scatter_div(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def scatter_update(self, *args, **kwargs):
        if False:
            return 10
        raise NotImplementedError