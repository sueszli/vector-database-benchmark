"""Class implementing utilities used by tf.distribute.Strategy."""
from collections import abc
import contextlib
import threading
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_values as tpu_values_lib
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.distribute.reduce_util import ReduceOp
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

@tf_export(v1=['distribute.get_loss_reduction'])
def get_loss_reduction():
    if False:
        return 10
    '`tf.distribute.ReduceOp` corresponding to the last loss reduction.\n\n  This is used to decide whether loss should be scaled in optimizer (used only\n  for estimator + v1 optimizer use case).\n\n  Returns:\n    `tf.distribute.ReduceOp` corresponding to the last loss reduction for\n    estimator and v1 optimizer use case. `tf.distribute.ReduceOp.SUM` otherwise.\n  '
    if not distribute_lib.get_strategy()._scale_loss_for_estimator:
        return ReduceOp.SUM
    last_reduction = ops.get_default_graph()._last_loss_reduction
    if last_reduction == losses_impl.Reduction.SUM or last_reduction == 'sum':
        return ReduceOp.SUM
    return ReduceOp.MEAN

def regroup(values, wrap_class=values_lib.PerReplica, always_wrap=False):
    if False:
        for i in range(10):
            print('nop')
    'Makes a nest per-replica into a nest of PerReplica/Mirrored values.\n\n  Args:\n    values: Values to regroup\n    wrap_class: Class that `values` be wrapped in.\n    always_wrap: Always wrap the `values` in `wrap_class` even if the values\n        are the same except for DistributeVariable.\n  Returns:\n    Wrapped `values`.\n  '
    v0 = values[0]
    if isinstance(v0, list):
        for v in values[1:]:
            assert isinstance(v, list)
            assert len(v) == len(v0), 'len(v) == %d, len(v0) == %d, v: %s, v0: %s' % (len(v), len(v0), v, v0)
        return [regroup(tuple((v[i] for v in values)), wrap_class, always_wrap) for i in range(len(v0))]
    if isinstance(v0, tuple):
        for v in values[1:]:
            assert isinstance(v, tuple)
            assert len(v) == len(v0), f'Values to regroup had different lengths: len(v) == {len(v)}, len(v0) == {len(v0)}, v: {v}, v0: {v0}'
        regrouped_tuple = tuple((regroup(tuple((v[i] for v in values)), wrap_class, always_wrap) for i in range(len(v0))))
        if hasattr(v0, '_fields'):
            assert hasattr(v0, '_make')
            return v0._make(regrouped_tuple)
        else:
            return regrouped_tuple
    if isinstance(v0, abc.Mapping):
        v0keys = v0.keys()
        for v in values[1:]:
            assert isinstance(v, abc.Mapping), 'v[0]: %r  v[i]: %r' % (v0, v)
            assert set(v.keys()) == set(v0keys), 'v[0].keys: %s  v[i].keys: %s' % (set(v0keys), set(v.keys()))
        return type(v0)({key: regroup(tuple((v[key] for v in values)), wrap_class, always_wrap) for key in v0keys})
    same_id = True
    for v in values[1:]:
        if v is not v0:
            same_id = False
            break
    if same_id and isinstance(v0, values_lib.DistributedVariable):
        return v0
    if same_id and (not always_wrap) and (value_container(v0) is v0):
        return v0
    if not isinstance(v0, resource_variable_ops._UnreadVariable) and value_container(v0) is not v0:
        assert not isinstance(v0, values_lib.MirroredVariable), 'ids = %s, values = %s' % ([id(v) for v in values], values)
        distributed_container = value_container(v0)
        assert distributed_container is not None
        for v in values[1:]:
            assert distributed_container is value_container(v)
        return distributed_container
    return wrap_class(values)

def select_replica(replica_id, structured):
    if False:
        print('Hello World!')
    'Specialize a nest of regular & per-replica values for one replica.'

    def _get(x):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(x, values_lib.DistributedVariable) or not isinstance(x, values_lib.DistributedValues):
            return x
        else:
            return x.values[replica_id]
    return nest.map_structure(_get, structured)

def select_replica_mirrored(replica_id, structured):
    if False:
        print('Hello World!')
    'Specialize a nest of regular & mirrored values for one replica.'
    assert_mirrored(structured)
    return select_replica(replica_id, structured)

def assert_mirrored(structured):
    if False:
        while True:
            i = 10
    'Raises if the structured is not composed of mirrored or regular values.'

    def _assert_mirrored(x):
        if False:
            return 10
        if isinstance(x, values_lib.DistributedValues) and (not is_mirrored(x)):
            raise TypeError('Expected value to be mirrored across replicas: %s in %s.' % (x, structured))
    nest.map_structure(_assert_mirrored, structured)

def update_regroup(extended, updates, group):
    if False:
        while True:
            i = 10
    'Regroup for an update, with dependencies to ensure all updates execute.'
    if not group:
        regrouped = regroup(updates, values_lib.Mirrored)
        return nest.map_structure(extended._local_results, regrouped)

    def _make_grouped_mirrored(values):
        if False:
            return 10
        'Convert per-replica list `values` into Mirrored type with grouping.'
        if len(values) == 1:
            return values_lib.Mirrored(values)
        g = control_flow_ops.group(values)
        if not all((tensor_util.is_tf_type(v) for v in values)):
            return g
        with_dep = []
        for v in values:
            with ops.device(v.device), ops.control_dependencies([g]):
                with_dep.append(array_ops.identity(v))
        return values_lib.Mirrored(with_dep)
    return regroup(updates, _make_grouped_mirrored)

def value_container(val):
    if False:
        for i in range(10):
            print('nop')
    'Returns the container that this per-replica `value` belongs to.\n\n  Args:\n    val: A value returned by `call_for_each_replica()` or a variable created in\n      `scope()`.\n\n  Returns:\n    A container that `value` belongs to.\n    If value does not belong to any container (including the case of\n    container having been destroyed), returns the value itself.\n  '
    container = None
    if not isinstance(val, values_lib.DistributedVariable):
        if hasattr(val, '_distributed_container'):
            container = val._distributed_container()
        elif isinstance(val, composite_tensor.CompositeTensor) and hasattr(val, 'handle') and hasattr(val.handle, '_distributed_container'):
            container = val.handle._distributed_container()
    return container if container is not None else val

def is_distributed_variable(v):
    if False:
        return 10
    'Determine if a variable is ds variable or TPU mirrored variable.'
    return getattr(v, 'is_distributed_variable', False)

def is_distributed_table(v):
    if False:
        while True:
            i = 10
    'Determine if an object is a DistributedTable.'
    return getattr(v, 'is_distributed_table', False)

def _validate_colocate_extended(v, extended):
    if False:
        for i in range(10):
            print('nop')
    variable_strategy = v._distribute_strategy
    if variable_strategy.extended is not extended:
        raise ValueError('`colocate_vars_with` must only be passed a variable created in this tf.distribute.Strategy.scope(), not %s created in scope: %s' % (v, variable_strategy))

def validate_colocate_distributed_variable(v, extended):
    if False:
        while True:
            i = 10
    if not isinstance(v, values_lib.DistributedVariable):
        raise ValueError('`colocate_vars_with` must only be passed a variable created in this tf.distribute.Strategy.scope(), not: %r' % (v,))
    _validate_colocate_extended(v, extended)

def validate_colocate(v, extended):
    if False:
        while True:
            i = 10
    if not hasattr(v, '_distribute_strategy'):
        raise ValueError('`colocate_vars_with` must only be passed a variable created in this tf.distribute.Strategy.scope(), not: %r' % (v,))
    _validate_colocate_extended(v, extended)

def _validate_synchronization(kwargs):
    if False:
        print('Hello World!')
    'Validate that given synchronization value is valid.'
    synchronization = kwargs.get('synchronization', vs.VariableSynchronization.AUTO)
    if synchronization == vs.VariableSynchronization.NONE:
        raise ValueError('`NONE` variable synchronization mode is not supported with tf.distribute strategy. Please change the `synchronization` for variable: ' + str(kwargs['name']))
    if synchronization not in (vs.VariableSynchronization.ON_READ, vs.VariableSynchronization.ON_WRITE, vs.VariableSynchronization.AUTO):
        raise ValueError('Invalid variable synchronization mode: %s for variable: %s' % (synchronization, kwargs['name']))
    if synchronization == vs.VariableSynchronization.AUTO:
        return vs.VariableSynchronization.ON_WRITE
    return synchronization

def _validate_aggregation(kwargs):
    if False:
        return 10
    aggregation = kwargs.get('aggregation', vs.VariableAggregation.NONE)
    if aggregation not in (vs.VariableAggregation.NONE, vs.VariableAggregation.SUM, vs.VariableAggregation.MEAN, vs.VariableAggregation.ONLY_FIRST_REPLICA):
        raise ValueError('Invalid variable aggregation mode: %s for variable: %s' % (aggregation, kwargs['name']))
    return aggregation

def create_mirrored_variable(strategy, real_mirrored_creator, class_mapping, policy_mapping, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Create distributed variables with given synchronization and aggregation.'
    if kwargs.pop('experimental_batch_initialization', None):
        variable_class_key = 'LazyVariableClass'
    else:
        variable_class_key = 'VariableClass'
    var_collections = kwargs.pop('collections', None)
    if var_collections is None:
        var_collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    kwargs['collections'] = []
    synchronization = _validate_synchronization(kwargs)
    kwargs['synchronization'] = synchronization
    aggregation = _validate_aggregation(kwargs)
    use_var_policy = getattr(strategy.extended, '_use_var_policy', False)
    kwargs.pop('caching_device', None)
    with record.stop_recording():
        value_list = real_mirrored_creator(**kwargs)
        for v in value_list:
            if hasattr(v, '_initializer_op') and v._initializer_op is None:
                v._initializer_op = control_flow_ops.no_op()
        if use_var_policy:
            var_policy_cls = policy_mapping.get(synchronization)
            var_policy = var_policy_cls(aggregation=aggregation)
            var_cls = class_mapping.get(variable_class_key)
            result = var_cls(strategy, value_list, aggregation, var_policy=var_policy)
        else:
            var_cls = class_mapping.get(synchronization)
            result = var_cls(strategy, value_list, aggregation)
    if not context.executing_eagerly():
        g = ops.get_default_graph()
        if kwargs.get('trainable', True):
            var_collections.append(ops.GraphKeys.TRAINABLE_VARIABLES)
            l = g.get_collection_ref(ops.GraphKeys.TRAINABLE_VARIABLES)
            for value in value_list:
                for (i, trainable_variable) in enumerate(l):
                    if value is trainable_variable:
                        del l[i]
                        break
        g.add_to_collections(var_collections, result)
    elif ops.GraphKeys.GLOBAL_STEP in var_collections:
        ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, result)
    return result

def is_mirrored(val):
    if False:
        while True:
            i = 10
    return getattr(val, '_is_mirrored', lambda : False)()

def is_sync_on_read(val):
    if False:
        i = 10
        return i + 15
    return not is_mirrored(val)

class CachingScopeLocal(threading.local):
    """Class for maintaining thread local state for caching scope."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(CachingScopeLocal, self).__init__()
        self.new_cache_scope_count = 0
        self.cache_scope_exited_count = 0

    def enter_scope(self):
        if False:
            return 10
        self.new_cache_scope_count += 1

    def exit_scope(self):
        if False:
            print('Hello World!')
        self.cache_scope_exited_count += 1

    def in_caching_scope(self):
        if False:
            return 10
        return self.new_cache_scope_count > self.cache_scope_exited_count
caching_scope_local = CachingScopeLocal()

@contextlib.contextmanager
def cache_variable_reads():
    if False:
        for i in range(10):
            print('nop')
    'Scope for caching variable reads for AggregatingVariable.\n\n  The variable reads for AggregatingVariable inside this scope are cached. i.e.\n  the first read of variable reads the value from possibly remote handle, but\n  subsequent reads are returned using local cached value.\n\n  For example:\n  strategy = ParameterServerStrategy...\n  with strategy.scope():\n    # Variable v is of AggregatingVariable type with actual variable residing\n    # on PS.\n    v = tf.Variable(1.0)\n\n  with distribute_utils.cache_variable_reads():\n    v.read_value()  # Reads value 1.0\n    v.assign(constant_op.constant(5.0))  # v changes to 5.0\n    t1 = v.read_value()\n    t2 = v.read_value()  # Both t1 & t2 return cached value 1.0 from local CPU.\n\n  Notes about cache_variable_reads scope:\n  1. Nesting of scope cache_variable_reads() is not supported\n  2. And when caching scope is enabled, the thread enabling the cache and\n    mirrored_run._MirroredReplicaThread threads spawned from it will have\n    caching enabled.\n\n  Yields:\n    A context for caching variables.\n  '
    try:
        if caching_scope_local.in_caching_scope():
            raise ValueError('cache_variable_reads scope cannot be nested')
        caching_scope_local.enter_scope()
        yield
    finally:
        caching_scope_local.exit_scope()
VARIABLE_POLICY_MAPPING = {vs.VariableSynchronization.ON_WRITE: values_lib.OnWritePolicy, vs.VariableSynchronization.ON_READ: values_lib.OnReadPolicy}
VARIABLE_CLASS_MAPPING = {'VariableClass': values_lib.DistributedVariable, vs.VariableSynchronization.ON_WRITE: values_lib.MirroredVariable, vs.VariableSynchronization.ON_READ: values_lib.SyncOnReadVariable}
TPU_VARIABLE_POLICY_MAPPING = {vs.VariableSynchronization.ON_WRITE: tpu_values_lib.TPUOnWritePolicy, vs.VariableSynchronization.ON_READ: tpu_values_lib.TPUOnReadPolicy}
TPU_VARIABLE_CLASS_MAPPING = {'VariableClass': tpu_values_lib.TPUDistributedVariable, 'LazyVariableClass': tpu_values_lib.TPULazyDistributedVariable, vs.VariableSynchronization.ON_WRITE: tpu_values_lib.TPUMirroredVariable, vs.VariableSynchronization.ON_READ: tpu_values_lib.TPUSyncOnReadVariable}