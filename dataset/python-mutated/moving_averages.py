"""Maintain moving averages of parameters."""
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.training import slot_creator
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls

@tf_export('__internal__.train.assign_moving_average', v1=[])
def assign_moving_average(variable, value, decay, zero_debias=True, name=None):
    if False:
        i = 10
        return i + 15
    "Compute the moving average of a variable.\n\n  The moving average of 'variable' updated with 'value' is:\n    variable * decay + value * (1 - decay)\n\n  The returned Operation sets 'variable' to the newly computed moving average,\n  by performing this subtraction:\n     variable -= (1 - decay) * (variable - value)\n\n  Since variables that are initialized to a `0` value will be `0` biased,\n  `zero_debias` optionally enables scaling by the mathematically correct\n  debiasing factor of\n    1 - decay ** num_updates\n  See Section 3 of (Kingma et al., 2015) for more details.\n\n  The names of the debias shadow variables, by default, include both the scope\n  they were created in and the scope of the variables they debias. They are also\n  given a uniquifying-suffix.\n\n  E.g.:\n\n  ```\n    with tf.compat.v1.variable_scope('scope1'):\n      with tf.compat.v1.variable_scope('scope2'):\n        var = tf.compat.v1.get_variable('foo')\n        update_1 = tf.assign_moving_average(var, 0.0, 1.0)\n        update_2 = tf.assign_moving_average(var, 0.0, 0.9)\n\n    # var.name: 'scope1/scope2/foo'\n    # shadow var names: 'scope1/scope2/scope1/scope2/foo/biased'\n    #                   'scope1/scope2/scope1/scope2/foo/biased_1'\n  ```\n\n  Args:\n    variable: A Variable.\n    value: A tensor with the same shape as 'variable'.\n    decay: A float `Tensor` or float value. The moving average decay.\n    zero_debias: A python bool. If true, assume the variable is 0-initialized\n      and unbias it, as in (Kingma et al., 2015). See docstring in\n        `_zero_debias` for more details.\n    name: Optional name of the returned operation.\n\n  Returns:\n    A tensor which if evaluated will compute and return the new moving average.\n\n  References:\n    Adam - A Method for Stochastic Optimization:\n      [Kingma et al., 2015](https://arxiv.org/abs/1412.6980)\n      ([pdf](https://arxiv.org/pdf/1412.6980.pdf))\n  "
    with ops.name_scope(name, 'AssignMovingAvg', [variable, value, decay]) as scope:
        decay = ops.convert_to_tensor(1.0 - decay, name='decay')
        if decay.dtype != variable.dtype.base_dtype:
            decay = math_ops.cast(decay, variable.dtype.base_dtype)

        def update_fn(v, value):
            if False:
                for i in range(10):
                    print('nop')
            return state_ops.assign_sub(v, (v - value) * decay, name=scope)

        def update(strategy, v, value):
            if False:
                return 10
            if zero_debias:
                return _zero_debias(strategy, v, value, decay)
            else:
                return _update(strategy, v, update_fn, args=(value,))
        replica_context = distribute_lib.get_replica_context()
        if replica_context:

            def merge_fn(strategy, v, value):
                if False:
                    while True:
                        i = 10
                value = strategy.extended.reduce_to(ds_reduce_util.ReduceOp.MEAN, value, v)
                return update(strategy, v, value)
            return replica_context.merge_call(merge_fn, args=(variable, value))
        else:
            strategy = distribute_lib.get_cross_replica_context()
            return update(strategy, variable, value)

def weighted_moving_average(value, decay, weight, truediv=True, collections=None, name=None):
    if False:
        print('Hello World!')
    'Compute the weighted moving average of `value`.\n\n  Conceptually, the weighted moving average is:\n    `moving_average(value * weight) / moving_average(weight)`,\n  where a moving average updates by the rule\n    `new_value = decay * old_value + (1 - decay) * update`\n  Internally, this Op keeps moving average variables of both `value * weight`\n  and `weight`.\n\n  Args:\n    value: A numeric `Tensor`.\n    decay: A float `Tensor` or float value. The moving average decay.\n    weight:  `Tensor` that keeps the current value of a weight. Shape should be\n      able to multiply `value`.\n    truediv:  Boolean, if `True`, dividing by `moving_average(weight)` is\n      floating point division.  If `False`, use division implied by dtypes.\n    collections:  List of graph collections keys to add the internal variables\n      `value * weight` and `weight` to. Defaults to\n      `[GraphKeys.GLOBAL_VARIABLES]`.\n    name: Optional name of the returned operation. Defaults to\n      "WeightedMovingAvg".\n\n  Returns:\n    An Operation that updates and returns the weighted moving average.\n  '
    if collections is None:
        collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    with variable_scope.variable_scope(name, 'WeightedMovingAvg', [value, weight, decay]) as scope:
        value_x_weight_var = variable_scope.get_variable('value_x_weight', shape=value.get_shape(), dtype=value.dtype, initializer=init_ops.zeros_initializer(), trainable=False, collections=collections)
        weight_var = variable_scope.get_variable('weight', shape=weight.get_shape(), dtype=weight.dtype, initializer=init_ops.zeros_initializer(), trainable=False, collections=collections)
        numerator = assign_moving_average(value_x_weight_var, value * weight, decay, zero_debias=False)
        denominator = assign_moving_average(weight_var, weight, decay, zero_debias=False)
        if truediv:
            return math_ops.truediv(numerator, denominator, name=scope.name)
        else:
            return math_ops.divide(numerator, denominator, name=scope.name)

def _update(strategy, var, update_fn, args):
    if False:
        print('Hello World!')
    'Applies updates depending on the context.'
    assert distribute_lib.in_cross_replica_context(), '_update can only be called in cross-replica context'
    if distribute_lib.get_update_replica_id() is not None:
        return update_fn(var, *args)
    else:
        return strategy.extended.update(var, update_fn, args)

def _zero_debias(strategy, unbiased_var, value, decay):
    if False:
        return 10
    'Compute the delta required for a debiased Variable.\n\n  All exponential moving averages initialized with Tensors are initialized to 0,\n  and therefore are biased to 0. Variables initialized to 0 and used as EMAs are\n  similarly biased. This function creates the debias updated amount according to\n  a scale factor, as in (Kingma et al., 2015).\n\n  To demonstrate the bias the results from 0-initialization, take an EMA that\n  was initialized to `0` with decay `b`. After `t` timesteps of seeing the\n  constant `c`, the variable have the following value:\n\n  ```\n    EMA = 0*b^(t) + c*(1 - b)*b^(t-1) + c*(1 - b)*b^(t-2) + ...\n        = c*(1 - b^t)\n  ```\n\n  To have the true value `c`, we would divide by the scale factor `1 - b^t`.\n\n  In order to perform debiasing, we use two shadow variables. One keeps track of\n  the biased estimate, and the other keeps track of the number of updates that\n  have occurred.\n\n  Args:\n    strategy: `Strategy` used to create and update variables.\n    unbiased_var: A Variable representing the current value of the unbiased EMA.\n    value: A Tensor representing the most recent value.\n    decay: A Tensor representing `1-decay` for the EMA.\n\n  Returns:\n    The amount that the unbiased variable should be updated. Computing this\n    tensor will also update the shadow variables appropriately.\n\n  References:\n    Adam - A Method for Stochastic Optimization:\n      [Kingma et al., 2015](https://arxiv.org/abs/1412.6980)\n      ([pdf](https://arxiv.org/pdf/1412.6980.pdf))\n\n  '
    with variable_scope.variable_scope(unbiased_var.name[:-len(':0')], values=[unbiased_var, value, decay]):
        with ops.init_scope():
            biased_initializer = init_ops.zeros_initializer()
            local_step_initializer = init_ops.zeros_initializer()

        def _maybe_get_unique(name):
            if False:
                return 10
            'Get name for a unique variable, if not `reuse=True`.'
            if variable_scope.get_variable_scope().reuse:
                return name
            vs_vars = [x.op.name for x in variable_scope.get_variable_scope().global_variables()]
            full_name = variable_scope.get_variable_scope().name + '/' + name
            if full_name not in vs_vars:
                return name
            idx = 1
            while full_name + '_%d' % idx in vs_vars:
                idx += 1
            return name + '_%d' % idx
        with strategy.extended.colocate_vars_with(unbiased_var):
            biased_var = variable_scope.get_variable(_maybe_get_unique('biased'), initializer=biased_initializer, shape=unbiased_var.get_shape(), dtype=unbiased_var.dtype, trainable=False)
            local_step = variable_scope.get_variable(_maybe_get_unique('local_step'), shape=[], dtype=unbiased_var.dtype, initializer=local_step_initializer, trainable=False)

    def update_fn(v, value, biased_var, local_step):
        if False:
            print('Hello World!')
        update_biased = state_ops.assign_sub(biased_var, (biased_var - value) * decay)
        update_local_step = local_step.assign_add(1)
        bias_factor = 1 - math_ops.pow(1.0 - decay, update_local_step)
        return state_ops.assign(v, update_biased / bias_factor, name=ops.get_name_scope() + '/')
    return _update(strategy, unbiased_var, update_fn, args=(value, biased_var, local_step))

@tf_export('train.ExponentialMovingAverage')
class ExponentialMovingAverage:
    """Maintains moving averages of variables by employing an exponential decay.

  When training a model, it is often beneficial to maintain moving averages of
  the trained parameters.  Evaluations that use averaged parameters sometimes
  produce significantly better results than the final trained values.

  The `apply()` method adds shadow copies of trained variables the first time
  it is called, and maintains a moving average of the trained variables in
  their shadow copies at every additional invocation.
  It should generally be called immediately after creating the model weights,
  and then after each training step.

  The `average()` method gives access to the shadow variables.
  It allows you to use the moving averages in place of the last trained values
  for evaluations, by loading the moving averages into your model via
  `var.assign(ema.average(var))`.
  Additionally, although `ExponentialMovingAverage`
  objects are not directly trackable by checkpoints,
  `average()` returns the moving average variables for your model weights,
  which you can then checkpoint. (There is an example
  of this near the bottom of this docstring).
  So, `average()` is useful when
  building an evaluation model, or when restoring a model from a checkpoint
  file.

  The moving averages are computed using exponential decay.  You specify the
  decay value (as a scalar float value, `Tensor`, or `Variable`) when creating
  the `ExponentialMovingAverage` object.  The shadow variables are initialized
  with the same initial values as the trained variables.  When you run `apply`
  to update the moving averages, each shadow variable is updated with the
  formula:

    `shadow_variable -= (1 - decay) * (shadow_variable - variable)`

  This is mathematically equivalent to the classic formula below, but the use
  of an `assign_sub` op (the `"-="` in the formula) allows concurrent lockless
  updates to the variables:

    `shadow_variable = decay * shadow_variable + (1 - decay) * variable`

  Reasonable values for `decay` are close to 1.0, typically in the
  multiple-nines range: 0.999, 0.9999, etc.

  To have fine-grained control over the value of the decay parameter during
  training, pass a scalar `tf.Variable` as the `decay` value to the constructor,
  and update the variable as needed.

  Example usage when creating a training model:

  ```python
  # Create variables.
  var0 = tf.Variable(...)
  var1 = tf.Variable(...)
  # ... use the variables to build a training model...

  # Create an ExponentialMovingAverage object
  ema = tf.train.ExponentialMovingAverage(decay=0.9999)

  # The first `apply` creates the shadow variables that hold the moving averages
  ema.apply([var0, var1])

  # grab the moving averages for checkpointing purposes or to be able to
  # load the moving averages into the model weights
  averages = [ema.average(var0), ema.average(var1)]

  ...
  def train_step(...):
  ...
    # Apply the optimizer.
    opt.minimize(my_loss, [var0, var1])

    # Update the moving averages
    # of var0 and var1 with additional calls to `apply`
    ema.apply([var0, var1])

  ...train the model by running train_step multiple times...
  ```

  There are several ways to use the moving averages for evaluations:

  1. Assign the values of the shadow variables to your model variables with
     `Variable.assign(...)` before evaluating your
     model. You can use the `average()`
     method to get the shadow variable for a given variable. To continue
     training after using this approach, make sure to record the unaveraged
     weights and restore them before continuing to train. You can see the
     tensorflow-addons' MovingAverage optimizer's `swap_weights` method for
     one example of how to swap variables efficiently in distributed settings:
     https://github.com/tensorflow/addons/blob/v0.13.0/tensorflow_addons/optimizers/moving_average.py#L151
  2. Make sure to checkpoint out your moving average variables in your
     `tf.train.Checkpoint`. At evaluation time, create your shadow variables and
     use `tf.train.Checkpoint` to restore the moving averages into the shadow
     variables. Then, load the moving averages into the actual model weights via
     `var.assign(moving_avg)`.
  3. Checkpoint out your moving average variables in your `tf.train.Checkpoint`.
     For evaluation, restore your model weights directly from the moving
     averages instead of from the non-averaged weights.
     Caution: If you choose this approach, include only the object-graph paths
     to the averaged path in your checkpoint restore.
     If you point both the unaveraged and averaged paths in a checkpoint
     restore to the same variables, it is hard to reason about whether your
     model will restore the averaged or non-averaged variables.

  Example of saving out then restoring the shadow variable values:

  ```python
  # Create variables.
  var0 = tf.Variable(...)
  var1 = tf.Variable(...)
  # ... use the variables to build a training model...

  # Create an ExponentialMovingAverage object, create the shadow variables,
  # and grab the moving averages for checkpointing purposes.
  # (The ExponentialMovingAverage object itself is not checkpointable)
  ema = tf.train.ExponentialMovingAverage(decay=0.9999)
  ema.apply([var0, var1])
  avg_var0 = ema.average(var0)
  avg_var1 = ema.average(var1)

  # Create a Checkpoint that will manage the model weights and the averages,
  checkpoint = tf.train.Checkpoint(model_weights=[var0, var1],
                                   averaged_weights=[avg_var0, avg_var1])
  ... # Do training

  # Save out the checkpoint including the model weights and the moving averages
  checkpoint.save(...)
  ```

  Restore option: restore all averaged & non-averaged weights, then load
  moving averages into the model via `var.assign()`
  ```python
  # Create variables.
  var0 = tf.Variable(...)
  var1 = tf.Variable(...)
  # ... use the variables to build a training model...

  # Create an ExponentialMovingAverage object, create the shadow variables,
  # and grab the moving averages for checkpoint restore purposes.
  # (The ExponentialMovingAverage object itself is not checkpointable)
  ema = tf.train.ExponentialMovingAverage(decay=0.9999)
  ema.apply([var0, var1])
  avg_var0 = ema.average(var0)
  avg_var1 = ema.average(var1)

  # Create a Checkpoint that will manage the model weights and the averages,
  checkpoint = tf.train.Checkpoint(model_weights=[var0, var1],
                                   averaged_weights=[avg_var0, avg_var1])
  checkpoint.restore(...)
  var0.assign(avg_var0)
  var1.assign(avg_var1)
  # var0 and var1 now hold the moving average values
  ```

  Restore option: Directly restore the moving averages into the model weights.
  ```python
  # Create variables.
  var0 = tf.Variable(...)
  var1 = tf.Variable(...)
  # ... use the variables to build a training model...

  # Create a Checkpoint that will manage two objects with trackable state,
  checkpoint = tf.train.Checkpoint(averaged_weights=[var0, var1])
  checkpoint.restore(...)
  # var0 and var1 now hold the moving average values
  ```
  """

    def __init__(self, decay, num_updates=None, zero_debias=False, name='ExponentialMovingAverage'):
        if False:
            while True:
                i = 10
        'Creates a new ExponentialMovingAverage object.\n\n    The `apply()` method has to be called to create shadow variables.\n    Follow-on calls to the `apply()` method will update the moving averages\n    in the shadow variables.\n    (In TF 1.x graphs `apply()` will return an update op to update\n    the moving averages which must be explicitly run).\n\n    The optional `num_updates` parameter allows one to tweak the decay rate\n    dynamically. It is typical to pass the count of training steps, usually\n    kept in a variable that is incremented at each step, in which case the\n    decay rate is lower at the start of training.  This makes moving averages\n    move faster.  If passed, the actual decay rate used is:\n\n      `min(decay, (1 + num_updates) / (10 + num_updates))`\n\n    Args:\n      decay: A scalar float value, `Tensor`, or `Variable`. The decay parameter.\n      num_updates: Optional count of number of updates applied to variables.\n      zero_debias: If `True`, zero debias moving-averages that are initialized\n        with tensors. (Note: moving averages may not be initialized with\n        non-variable tensors when eager execution is enabled).\n      name: String. Optional prefix name to use for the name of ops added in\n        `apply()`.\n    '
        self._decay = decay
        self._num_updates = num_updates
        self._zero_debias = zero_debias
        self._name = name
        self._averages = {}

    @property
    def name(self):
        if False:
            while True:
                i = 10
        'The name of this ExponentialMovingAverage object.'
        return self._name

    def apply(self, var_list=None):
        if False:
            while True:
                i = 10
        "Maintains moving averages of variables.\n\n    `var_list` must be a list of `Variable` objects.  This method\n    creates shadow variables (holding the moving averages)\n    for all elements of `var_list`, and\n    updates the moving averages using the current `var_list` values. Shadow\n    variables for `Variable` objects are initialized to the variable's initial\n    value.\n\n    Shadow variables are created with `trainable=False`. To access them you\n    can use the EMA object's `average` method. Note that `EMA` objects are\n    not trackable by checkpoints, so if you want to checkpoint or restore the\n    moving variables you will need to manually grab the shadow\n    variables via `average()` and assign them as `tf.Module` properties or\n    directly pass them to your `tf.train.Checkpoint`.\n\n    Note that `apply()` can be called multiple times. When eager execution is\n    enabled each call to apply will update the variables once, so this needs to\n    be called in a loop.\n\n    In legacy TF 1.x graphs, this method returns an op that updates all\n    shadow variables from the current value of their associated variables. In\n    TF 1.x graphs without automatically control dependencies this op needs to be\n    manually run.\n\n    Args:\n      var_list: A list of Variable objects. The variables\n        must be of types bfloat16, float16, float32, or float64.\n        (In legacy TF 1.x graphs these may be tensors, but this is unsupported\n        when eager execution is enabled.)\n\n    Returns:\n      An Operation that updates the moving averages.\n\n    Raises:\n      TypeError: If the arguments are not an allowed type.\n    "
        if var_list is None:
            var_list = variables.trainable_variables()
        for v in var_list:
            if isinstance(v, tensor.Tensor) and ops.executing_eagerly_outside_functions():
                raise TypeError('tf.train.ExponentialMovingAverage does not support non-Variable tensors when eager execution is enabled.')
        zero_debias_true = set()
        for var in var_list:
            if var.dtype.base_dtype not in [dtypes.bfloat16, dtypes.float16, dtypes.float32, dtypes.float64]:
                raise TypeError('The variables must be half, float, or double: %s' % var.name)
            if var.ref() not in self._averages:
                with ops.init_scope():
                    if isinstance(var, variables.Variable):
                        with ops.device(var.device):
                            initialized_value = cond.cond(variable_v1.is_variable_initialized(var), var.read_value, lambda : var.initial_value)
                        avg = slot_creator.create_slot(var, initialized_value, self.name, colocate_with_primary=True, copy_xla_sharding=True)
                        ops.add_to_collection(ops.GraphKeys.MOVING_AVERAGE_VARIABLES, var)
                    else:
                        avg = slot_creator.create_zeros_slot(var, self.name, colocate_with_primary=var.op.type in ['Variable', 'VariableV2', 'VarHandleOp'], copy_xla_sharding=True)
                        if self._zero_debias:
                            zero_debias_true.add(avg.ref())
                self._averages[var.ref()] = avg
        with ops.name_scope(self.name) as scope:
            decay = ops.convert_to_tensor(self._decay, dtype=dtypes.float32, name='decay')
            if self._num_updates is not None:
                num_updates = math_ops.cast(self._num_updates, dtypes.float32, name='num_updates')
                decay = math_ops.minimum(decay, (1.0 + num_updates) / (10.0 + num_updates))
            updates = []
            for var in var_list:
                avg = self._averages[var.ref()]
                zero_debias = avg.ref() in zero_debias_true
                updates.append(assign_moving_average(avg, var, decay, zero_debias))
            return control_flow_ops.group(*updates, name=scope)

    def average(self, var):
        if False:
            i = 10
            return i + 15
        'Returns the `Variable` holding the average of `var`.\n\n    Args:\n      var: A `Variable` object.\n\n    Returns:\n      A `Variable` object or `None` if the moving average of `var`\n      is not maintained.\n    '
        return self._averages.get(var.ref(), None)

    @doc_controls.do_not_generate_docs
    def average_name(self, var):
        if False:
            while True:
                i = 10
        '[Meant for TF1] Returns name of `Variable` holding the average for `var`.\n\n    (Designed to work with legacy `tf.compat.v1.train.Saver`, it is sensitive to\n    specific variable names and not recommended for TF2)\n\n    The typical scenario for `ExponentialMovingAverage` is to compute moving\n    averages of variables during training, and restore the variables from the\n    computed moving averages during evaluations.\n\n    To restore variables, you have to know the name of the shadow variables.\n    That name and the original variable can then be passed to a `Saver()` object\n    to restore the variable from the moving average value with:\n      `saver = tf.compat.v1.train.Saver({ema.average_name(var): var})`\n\n    `average_name()` can be called whether or not `apply()` has been called.\n\n    Args:\n      var: A `Variable` object.\n\n    Returns:\n      A string: The name of the variable that will be used or was used\n      by the `ExponentialMovingAverage class` to hold the moving average of\n      `var`.\n    '
        if var.ref() in self._averages:
            return self._averages[var.ref()].name[:-len(':0')]
        return ops.get_default_graph().unique_name(var.name[:-len(':0')] + '/' + self.name, mark_as_used=False)

    @doc_controls.do_not_generate_docs
    def variables_to_restore(self, moving_avg_variables=None):
        if False:
            i = 10
            return i + 15
        '[Designed for TF 1.x] Returns a map of names to `Variables` to restore.\n\n    (Designed to work with legacy `tf.compat.v1.train.Saver`, sensitive to\n    specific variable names and not recommended for TF2)\n\n    If a variable has a moving average, use the moving average variable name as\n    the restore name; otherwise, use the variable name.\n\n    For example,\n\n    ```python\n      variables_to_restore = ema.variables_to_restore()\n      saver = tf.compat.v1.train.Saver(variables_to_restore)\n    ```\n\n    Below is an example of such mapping:\n\n    ```\n      conv/batchnorm/gamma/ExponentialMovingAverage: conv/batchnorm/gamma,\n      conv_4/conv2d_params/ExponentialMovingAverage: conv_4/conv2d_params,\n      global_step: global_step\n    ```\n\n    Args:\n      moving_avg_variables: a list of variables that require to use of the\n        moving average variable name to be restored. If None, it will default to\n        variables.moving_average_variables() + variables.trainable_variables()\n\n    Returns:\n      A map from restore_names to variables. The restore_name is either the\n      original or the moving average version of the variable name, depending\n      on whether the variable name is in the `moving_avg_variables`.\n    '
        name_map = {}
        if moving_avg_variables is None:
            moving_avg_variables = variables.trainable_variables()
            moving_avg_variables += variables.moving_average_variables()
        moving_avg_variables = set((v.ref() for v in moving_avg_variables))
        for v in moving_avg_variables:
            name_map[self.average_name(v.deref())] = v.deref()
        moving_avg_variable_names = set((v.deref().name for v in moving_avg_variables))
        for v in list(set(variables.global_variables())):
            if v.name not in moving_avg_variable_names and v.op.name not in name_map:
                name_map[v.op.name] = v
        return name_map