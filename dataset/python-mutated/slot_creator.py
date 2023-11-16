"""Standard functions for creating slots.

A slot is a `Variable` created with the same first m-dimension as a primary
variable or `Tensor`. A slot is always scoped in the namespace of the primary
object and typically has the same device and type.

Slots are typically used as accumulators to track values associated with
the primary object:

```python
# Optimizers can create a slot for each variable to track accumulators
accumulators = {var : create_zeros_slot(var, "momentum") for var in vs}
for var in vs:
  apply_momentum(var, accumulators[var], lr, grad, momentum_tensor)

# Slots can also be used for moving averages
mavg = create_slot(var, var.initialized_value(), "exponential_moving_avg")
update_mavg = mavg.assign_sub((mavg - var) * (1 - decay))
```
"""
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables

def _create_slot_var(primary, val, scope, validate_shape, shape, dtype, *, copy_xla_sharding=False):
    if False:
        while True:
            i = 10
    'Helper function for creating a slot variable.'
    current_partitioner = variable_scope.get_variable_scope().partitioner
    variable_scope.get_variable_scope().set_partitioner(None)
    shape = shape if callable(val) else None
    if resource_variable_ops.is_resource_variable(primary):
        use_resource = True
    elif isinstance(primary, ref_variable.RefVariable):
        use_resource = False
    else:
        use_resource = None
    slot = variable_scope.get_variable(scope, initializer=val, trainable=False, use_resource=use_resource, shape=shape, dtype=dtype, validate_shape=validate_shape)
    variable_scope.get_variable_scope().set_partitioner(current_partitioner)
    if isinstance(primary, variables.Variable) and primary._save_slice_info:
        real_slot_name = slot.name[len(primary.op.name + '/'):-2]
        slice_info = primary._save_slice_info
        n = slot.shape.ndims
        if n is None or n > 0:
            slot._set_save_slice_info(variables.Variable.SaveSliceInfo(slice_info.full_name + '/' + real_slot_name, slice_info.full_shape[:n], slice_info.var_offset[:n], slice_info.var_shape[:n]))

    def _has_same_rank(primary_shape, slot_shape):
        if False:
            return 10
        return primary_shape.rank is not None and slot_shape.rank is not None and (primary_shape.rank == slot_shape.rank)
    if copy_xla_sharding and _has_same_rank(primary.shape, slot.shape):
        slot = xla_sharding.copy_sharding(primary, slot, use_sharding_op=False)
    return slot

def create_slot(primary, val, name, colocate_with_primary=True, *, copy_xla_sharding=False):
    if False:
        while True:
            i = 10
    'Create a slot initialized to the given value.\n\n  The type of the slot is determined by the given value.\n\n  Args:\n    primary: The primary `Variable` or `Tensor`.\n    val: A `Tensor` specifying the initial value of the slot.\n    name: Name to use for the slot variable.\n    colocate_with_primary: Boolean.  If True the slot is located\n      on the same device as `primary`.\n    copy_xla_sharding: Boolean. If True also copies XLA sharding\n      from primary.\n\n  Returns:\n    A `Variable` object.\n  '
    validate_shape = val.get_shape().is_fully_defined()
    if isinstance(primary, variables.Variable):
        prefix = primary._shared_name
    else:
        prefix = primary.op.name
    with variable_scope.variable_scope(None, prefix + '/' + name):
        if colocate_with_primary:
            distribution_strategy = distribute_lib.get_strategy()
            with distribution_strategy.extended.colocate_vars_with(primary):
                return _create_slot_var(primary, val, '', validate_shape, None, None, copy_xla_sharding=copy_xla_sharding)
        else:
            return _create_slot_var(primary, val, '', validate_shape, None, None, copy_xla_sharding=copy_xla_sharding)

def create_slot_with_initializer(primary, initializer, shape, dtype, name, colocate_with_primary=True, *, copy_xla_sharding=False):
    if False:
        while True:
            i = 10
    'Creates a slot initialized using an `Initializer`.\n\n  The type of the slot is determined by the given value.\n\n  Args:\n    primary: The primary `Variable` or `Tensor`.\n    initializer: An `Initializer`.  The initial value of the slot.\n    shape: Shape of the initial value of the slot.\n    dtype: Type of the value of the slot.\n    name: Name to use for the slot variable.\n    colocate_with_primary: Boolean.  If True the slot is located\n      on the same device as `primary`.\n    copy_xla_sharding: Boolean. If True also copies XLA sharding\n      from primary.\n\n  Returns:\n    A `Variable` object.\n  '
    validate_shape = shape.is_fully_defined()
    if isinstance(primary, variables.Variable):
        prefix = primary._shared_name
    else:
        prefix = primary.op.name
    with variable_scope.variable_scope(None, prefix + '/' + name):
        if colocate_with_primary:
            distribution_strategy = distribute_lib.get_strategy()
            with distribution_strategy.extended.colocate_vars_with(primary):
                return _create_slot_var(primary, initializer, '', validate_shape, shape, dtype, copy_xla_sharding=copy_xla_sharding)
        else:
            return _create_slot_var(primary, initializer, '', validate_shape, shape, dtype, copy_xla_sharding=copy_xla_sharding)

def create_zeros_slot(primary, name, dtype=None, colocate_with_primary=True, *, copy_xla_sharding=False):
    if False:
        return 10
    'Create a slot initialized to 0 with same shape as the primary object.\n\n  Args:\n    primary: The primary `Variable` or `Tensor`.\n    name: Name to use for the slot variable.\n    dtype: Type of the slot variable.  Defaults to the type of `primary`.\n    colocate_with_primary: Boolean.  If True the slot is located\n      on the same device as `primary`.\n    copy_xla_sharding: Boolean. If True also copies XLA sharding\n      from primary.\n\n  Returns:\n    A `Variable` object.\n  '
    if dtype is None:
        dtype = primary.dtype
    slot_shape = primary.get_shape()
    if slot_shape.is_fully_defined():
        initializer = init_ops.zeros_initializer()
        return create_slot_with_initializer(primary, initializer, slot_shape, dtype, name, colocate_with_primary=colocate_with_primary, copy_xla_sharding=copy_xla_sharding)
    else:
        if isinstance(primary, variables.Variable):
            slot_shape = array_ops.shape(cond.cond(variable_v1.is_variable_initialized(primary), primary.read_value, lambda : primary.initial_value))
        else:
            slot_shape = array_ops.shape(primary)
        val = array_ops.zeros(slot_shape, dtype=dtype)
        return create_slot(primary, val, name, colocate_with_primary=colocate_with_primary, copy_xla_sharding=copy_xla_sharding)