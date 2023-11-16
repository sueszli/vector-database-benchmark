"""Utility functions used by values.py and ps_values.py."""
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options
from tensorflow.python.training.saving import saveable_object

def write_object_proto(var, proto, options):
    if False:
        for i in range(10):
            print('nop')
    'Update a SavedObject proto for the caller.\n\n  If a DistributedVariable object supports this method, it will be called when\n  saving with a pre-built `SavedObject` proto representing the object, plus an\n  instance of `SaveOptions`. This method is then free to modify that proto\n  instance.\n\n  `DistributedVariable` with `AUTO` or `ON_WRITE` synchronization optionally\n   write out information about their components to the\n   `experimental_distributed_variable_components` field of a\n   `SavedVariable` (depending on the `SaveOptions` variable policy).\n\n  Args:\n    var: The DistributedVariable object.\n    proto: A pre-built `SavedObject` proto for this object. It is assumed this\n      will be a `SavedVariable` instance.\n    options: A `SaveOptions` instance.\n  '
    if options.experimental_variable_policy._expand_distributed_variables():
        for var in var.values:
            var_proto = proto.variable.experimental_distributed_variable_components.add()
            var_proto.name = var.name.split(':')[0]
            var_proto.device = var.device

def get_on_write_saveable(var, primary_var, name):
    if False:
        i = 10
        return i + 15
    'Return saveable spec for AUTO and ON_WRITE variables.'

    def tensor():
        if False:
            i = 10
            return i + 15
        if context.executing_eagerly() and (not primary_var.is_initialized()):
            return None
        strategy = var.distribute_strategy
        return strategy.extended.read_var(var)
    spec = saveable_object.SaveSpec(tensor=tensor, slice_spec='', name=name, dtype=var.dtype, device=primary_var.device)
    return (tensor, [spec])

def get_on_write_restore_ops(var, tensor):
    if False:
        print('Hello World!')
    'Return restore ops for AUTO and ON_WRITE variables.'
    packed_var = var._packed_variable
    if packed_var is not None:
        return control_flow_ops.group(tuple((assign_on_device(d, packed_var, tensor) for d in packed_var.devices)))
    return control_flow_ops.group(tuple((assign_on_device(v.device, v, tensor) for v in var.values)))

def get_on_read_saveable(var, primary_var, name):
    if False:
        i = 10
        return i + 15
    'Return saveables for ON_READ variable.'

    def tensor():
        if False:
            i = 10
            return i + 15
        return var._get_cross_replica()
    spec = saveable_object.SaveSpec(tensor=tensor, slice_spec='', name=name, dtype=var.dtype, device=primary_var.device)
    return (tensor, [spec])

def get_on_read_restore_ops(var, tensor, aggregation):
    if False:
        return 10
    'Return restore ops for ON_READ variables.'
    if aggregation == vs.VariableAggregation.SUM:
        strategy = var.distribute_strategy
        tensor = math_ops.cast(tensor / strategy.num_replicas_in_sync, var.dtype)
    return control_flow_ops.group(tuple((assign_on_device(v.device, v, tensor) for v in var.values)))

def in_replica_update_context():
    if False:
        i = 10
        return i + 15
    return distribute_lib.get_update_replica_id() is not None

def on_write_assign(var, value, use_locking=False, name=None, read_value=True):
    if False:
        i = 10
        return i + 15
    assign_fn = lambda var, *a, **kw: var.assign(*a, **kw)
    return var._update(update_fn=assign_fn, value=value, use_locking=use_locking, name=name, read_value=read_value)

def on_write_assign_add(var, value, use_locking=False, name=None, read_value=True):
    if False:
        for i in range(10):
            print('nop')
    assign_add_fn = lambda var, *a, **kw: var.assign_add(*a, **kw)
    return var._update(update_fn=assign_add_fn, value=value, use_locking=use_locking, name=name, read_value=read_value)

def on_write_assign_sub(var, value, use_locking=False, name=None, read_value=True):
    if False:
        while True:
            i = 10
    assign_sub_fn = lambda var, *a, **kw: var.assign_sub(*a, **kw)
    return var._update(update_fn=assign_sub_fn, value=value, use_locking=use_locking, name=name, read_value=read_value)

def assign_on_each_device(var, assign_func, value, read_value):
    if False:
        while True:
            i = 10
    'Update the variable on each replica with the given assign_func and value.'
    if var._packed_variable is not None:
        update = control_flow_ops.group(tuple((assign_func(d, var._packed_variable, value) for d in var._devices)))
    else:
        update = control_flow_ops.group(tuple((assign_func(v.device, v, value) for v in var._values)))
    if not read_value:
        return update
    with ops.control_dependencies([update] if update else []):
        return var.read_value()

def on_read_assign_sub_cross_replica(var, value, read_value=True):
    if False:
        print('Hello World!')
    with distribute_lib.enter_or_assert_strategy(var.distribute_strategy):
        if distribute_lib.in_cross_replica_context():
            if var.aggregation == vs.VariableAggregation.SUM:
                raise ValueError('SyncOnReadVariable does not support `assign_sub` in cross-replica context when aggregation is set to `tf.VariableAggregation.SUM`.')
            return assign_on_each_device(var, assign_sub_on_device, value, read_value)

def on_read_assign_add_cross_replica(var, value, read_value=True):
    if False:
        return 10
    with distribute_lib.enter_or_assert_strategy(var.distribute_strategy):
        if distribute_lib.in_cross_replica_context():
            if var.aggregation == vs.VariableAggregation.SUM:
                raise ValueError('SyncOnReadVariable does not support `assign_add` in cross-replica context when aggregation is set to `tf.VariableAggregation.SUM`.')
            return assign_on_each_device(var, assign_add_on_device, value, read_value)

def on_read_assign_cross_replica(var, value, read_value=True):
    if False:
        for i in range(10):
            print('nop')
    'Return the value of the variable in cross replica context.'
    with distribute_lib.enter_or_assert_strategy(var.distribute_strategy):
        if distribute_lib.in_cross_replica_context():
            tensor = value
            if var.aggregation == vs.VariableAggregation.SUM:
                strategy = var._distribute_strategy
                tensor = math_ops.cast(tensor / strategy.num_replicas_in_sync, var.dtype)
            return assign_on_each_device(var, assign_on_device, tensor, read_value)

def scatter_sub(var, sparse_delta, use_locking=False, name=None):
    if False:
        for i in range(10):
            print('nop')
    scatter_sub_fn = lambda var, *a, **kw: var.scatter_sub(*a, **kw)
    return var._update(update_fn=scatter_sub_fn, value=sparse_delta, use_locking=use_locking, name=name)

def scatter_add(var, sparse_delta, use_locking=False, name=None):
    if False:
        i = 10
        return i + 15
    scatter_add_fn = lambda var, *a, **kw: var.scatter_add(*a, **kw)
    return var._update(update_fn=scatter_add_fn, value=sparse_delta, use_locking=use_locking, name=name)

def scatter_mul(var, sparse_delta, use_locking=False, name=None):
    if False:
        while True:
            i = 10
    scatter_mul_fn = lambda var, *a, **kw: var.scatter_mul(*a, **kw)
    return var._update(update_fn=scatter_mul_fn, value=sparse_delta, use_locking=use_locking, name=name)

def scatter_div(var, sparse_delta, use_locking=False, name=None):
    if False:
        i = 10
        return i + 15
    scatter_div_fn = lambda var, *a, **kw: var.scatter_div(*a, **kw)
    return var._update(update_fn=scatter_div_fn, value=sparse_delta, use_locking=use_locking, name=name)

def scatter_min(var, sparse_delta, use_locking=False, name=None):
    if False:
        return 10
    scatter_min_fn = lambda var, *a, **kw: var.scatter_min(*a, **kw)
    return var._update(update_fn=scatter_min_fn, value=sparse_delta, use_locking=use_locking, name=name)

def scatter_max(var, sparse_delta, use_locking=False, name=None):
    if False:
        while True:
            i = 10
    scatter_max_fn = lambda var, *a, **kw: var.scatter_max(*a, **kw)
    return var._update(update_fn=scatter_max_fn, value=sparse_delta, use_locking=use_locking, name=name)

def scatter_update(var, sparse_delta, use_locking=False, name=None):
    if False:
        for i in range(10):
            print('nop')
    scatter_update_fn = lambda var, *a, **kw: var.scatter_update(*a, **kw)
    return var._update(update_fn=scatter_update_fn, value=sparse_delta, use_locking=use_locking, name=name)

def get_current_replica_id_as_int():
    if False:
        i = 10
        return i + 15
    'Returns the current replica ID as an integer, or `None`.'
    replica_context = distribute_lib.get_replica_context()
    if replica_context:
        replica_id = replica_context._replica_id
        if not isinstance(replica_id, int):
            replica_id = tensor_util.constant_value(replica_id)
    else:
        replica_id = distribute_lib.get_update_replica_id()
    return replica_id

def assign_on_device(device, variable, tensor):
    if False:
        print('Hello World!')
    with ops.device(device):
        return variable.assign(tensor)

def assign_add_on_device(device, variable, tensor):
    if False:
        return 10
    with ops.device(device):
        return variable.assign_add(tensor)

def assign_sub_on_device(device, variable, tensor):
    if False:
        while True:
            i = 10
    with ops.device(device):
        return variable.assign_sub(tensor)

def assert_replica_context(strategy):
    if False:
        for i in range(10):
            print('nop')
    replica_context = distribute_lib.get_replica_context()
    if not replica_context:
        raise RuntimeError('Replica-local variables may only be assigned in a replica context.')
    if replica_context.strategy is not strategy:
        raise RuntimeError('Replica-local variables may only be assigned in a replica context.')

def apply_aggregation(strategy, value, aggregation, destinations):
    if False:
        print('Hello World!')
    if aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
        return strategy.extended.broadcast_to(strategy.experimental_local_results(value)[0], destinations=destinations)
    reduce_op = reduce_util.ReduceOp.from_variable_aggregation(aggregation)
    return strategy.extended.reduce_to(reduce_op, value, destinations)
aggregation_error_msg = 'You must specify an aggregation method to update a {variable_type} in Replica Context. You can do so by passing an explicit value for argument `aggregation` to tf.Variable(..).e.g. `tf.Variable(..., aggregation=tf.VariableAggregation.SUM)``tf.VariableAggregation` lists the possible aggregation methods.This is required because {variable_type} should always be kept in sync. When updating them or assigning to them in a replica context, we automatically try to aggregate the values before updating the variable. For this aggregation, we need to know the aggregation method. Another alternative is to not try to update such {variable_type} in replica context, but in cross replica context. You can enter cross replica context by calling `tf.distribute.get_replica_context().merge_call(merge_fn, ..)`.Inside `merge_fn`, you can then update the {variable_type} using `tf.distribute.StrategyExtended.update()`.'
scatter_error_msg = '{op_name} is only supported for mirrored variable (variable created within certain `tf.distribute.Strategy` scope) with NONE or `ONLY_FIRST_REPLICA` aggregation, got: {aggregation}.'

def is_saving_non_distributed():
    if False:
        print('Hello World!')
    "Returns whether we're saving a non-distributed version of the model.\n\n  It returns True iff we are in saving context and are saving a non-distributed\n  version of the model. That is, SaveOptions.experimental_variable_policy is\n  NONE.\n\n  Returns:\n    A boolean.\n  "
    if not save_context.in_save_context():
        return False
    options = save_context.get_save_options()
    return options.experimental_variable_policy != save_options.VariablePolicy.EXPAND_DISTRIBUTED_VARIABLES

def mark_as_unsaveable():
    if False:
        for i in range(10):
            print('nop')
    'Marks the function as unsaveable if not inside save context.'
    if ops.inside_function() and (not save_context.in_save_context()):
        ops.get_default_graph().mark_as_unsaveable("\nConcreteFunction that uses distributed variables in certain way cannot be saved.\nIf you're saving with\n\ntf.saved_model.save(..., signatures=f.get_concrete_function())\n\ndo\n\n@tf.function(input_signature=...)\ndef f_with_input_signature():\n  ...\n\ntf.saved_model.save(..., signatures=f_with_input_signature)`\n\ninstead.")