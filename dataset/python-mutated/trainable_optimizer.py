"""A base class definition for trainable optimizers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import itertools
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
OPTIMIZER_SCOPE = 'LOL'
_LOCAL_VARIABLE_PREFIX = 'local_state_'
_LOCAL_STATE_VARIABLE_COLLECTION = 'local_state_collection'
EPSILON = 1e-06

class TrainableOptimizer(tf.train.Optimizer):
    """Base class for trainable optimizers.

  A trainable optimizer is an optimizer that has parameters that can themselves
  be learned (meta-optimized).

  Subclasses must implement:
      _compute_update(self, param, grad, state)
  """

    def __init__(self, name, state_keys, use_attention=False, use_log_objective=False, obj_train_max_multiplier=-1, use_second_derivatives=True, use_numerator_epsilon=False, **kwargs):
        if False:
            print('Hello World!')
        'Initializes the optimizer with the given name and settings.\n\n    Args:\n      name: The name string for this optimizer.\n      state_keys: The names of any required state variables (list)\n      use_attention: Whether this optimizer uses attention (Default: True)\n      use_log_objective: Whether this optimizer uses the logarithm of the\n          objective when computing the loss (Default: False)\n      obj_train_max_multiplier: The maximum multiplier for the increase in the\n          objective before meta-training is stopped. If <= 0, meta-training is\n          not stopped early. (Default: -1)\n      use_second_derivatives: Whether this optimizer uses second derivatives in\n          meta-training. This should be set to False if some second derivatives\n          in the meta-training problem set are not defined in Tensorflow.\n          (Default: True)\n      use_numerator_epsilon: Whether to use epsilon in the numerator when\n          scaling the problem objective during meta-training. (Default: False)\n      **kwargs: Any additional keyword arguments.\n    '
        self.use_second_derivatives = use_second_derivatives
        self.state_keys = sorted(state_keys)
        self.use_attention = use_attention
        self.use_log_objective = use_log_objective
        self.obj_train_max_multiplier = obj_train_max_multiplier
        self.use_numerator_epsilon = use_numerator_epsilon
        use_locking = False
        super(TrainableOptimizer, self).__init__(use_locking, name)

    def _create_slots(self, var_list):
        if False:
            return 10
        'Creates all slots needed by the variables.\n\n    Args:\n      var_list: A list of `Variable` objects.\n    '
        for var in var_list:
            init_states = self._initialize_state(var)
            for slot_name in sorted(init_states):
                slot_var_name = '{}_{}'.format(self.get_name(), slot_name)
                value = init_states[slot_name]
                self._get_or_make_slot(var, value, slot_name, slot_var_name)

    def _initialize_state(self, var):
        if False:
            i = 10
            return i + 15
        'Initializes any state required for this variable.\n\n    Args:\n      var: a tensor containing parameters to be optimized\n\n    Returns:\n      state: a dictionary mapping state keys to initial state values (tensors)\n    '
        return {}

    def _initialize_global_state(self):
        if False:
            while True:
                i = 10
        'Initializes any global state values.'
        return []

    def _apply_common(self, grad, var):
        if False:
            i = 10
            return i + 15
        'Applies the optimizer updates to the variables.\n\n    Note: this should only get called via _apply_dense or _apply_sparse when\n    using the optimizer via optimizer.minimize or optimizer.apply_gradients.\n    During meta-training, the optimizer.train function should be used to\n    construct an optimization path that is differentiable.\n\n    Args:\n      grad: A tensor representing the gradient.\n      var: A tf.Variable with the same shape as grad.\n\n    Returns:\n      update_op: A tensorflow op that assigns new values to the variable, and\n          also defines dependencies that update the state variables for the\n          optimizer.\n    '
        state = {key: self.get_slot(var, key) for key in self.get_slot_names()}
        (new_var, new_state) = self._compute_update(var, grad, state)
        state_assign_ops = [tf.assign(state_var, new_state[key]) for (key, state_var) in state.items()]
        with tf.control_dependencies(state_assign_ops):
            update_op = var.assign(new_var)
        return update_op

    def _apply_dense(self, grad, var):
        if False:
            for i in range(10):
                print('nop')
        "Adds ops to apply dense gradients to 'var'."
        return self._apply_common(grad, var)

    def _apply_sparse(self, grad, var):
        if False:
            for i in range(10):
                print('nop')
        "Adds ops to apply sparse gradients to 'var'."
        return self._apply_common(grad, var)

    def _compute_update(self, param, grad, state):
        if False:
            while True:
                i = 10
        'Computes the update step for optimization.\n\n    Args:\n      param: A tensor of parameters to optimize.\n      grad: The gradient tensor of the objective with respect to the parameters.\n          (It has the same shape as param.)\n      state: A dictionary containing any extra state required by the optimizer.\n\n    Returns:\n      updated_params: The updated parameters.\n      updated_state: The dictionary of updated state variable(s).\n    '
        raise NotImplementedError

    def _compute_updates(self, params, grads, states, global_state):
        if False:
            print('Hello World!')
        'Maps the compute update functions for each parameter.\n\n    This function can be overriden by a subclass if the subclass wants to\n    combine information across the different parameters in the list.\n\n    Args:\n      params: A list of parameter tensors.\n      grads: A list of gradients corresponding to each parameter.\n      states: A list of state variables corresponding to each parameter.\n      global_state: A list of global state variables for the problem.\n\n    Returns:\n      new_params: The updated parameters.\n      new_states: The updated states.\n      new_global_state: The updated global state.\n      attention_params: A list of attention parameters. This is the same as\n          new_params if the optimizer does not use attention.\n    '
        args = zip(params, grads, states)
        (new_params, new_states) = zip(*list(itertools.starmap(self._compute_update, args)))
        return (list(new_params), list(new_states), global_state, list(new_params))

    def train(self, problem, dataset):
        if False:
            return 10
        'Creates graph operations to train the optimizer.\n\n    Args:\n      problem: A problem_generator.Problem instance to train on.\n      dataset: A datasets.Dataset tuple to use when training.\n\n    Returns:\n      meta_objective: A tensorflow operation for computing the meta-objective\n      obj_weights: A tensor placeholder for feeding in the objective weights\n      obj_values: The subproblem objective values during optimization\n      batches: The batch indexes tensor for overriding with feed_dict\n      first_unroll: A placeholder signifying if this is a first unroll\n        (this will propagate the gradients slightly differently).\n      reset_state: A placeholder signifying that the rnn state should be reset.\n      output_state: The final state of the optimizer\n      init_loop_vars_to_override: Local variables that can be assigned to\n        propagate the optimizer and problem state for unrolling\n      final_loop_vals: Final values of the loop variables that can be\n        assigned to init_loop_vars_to_override.\n    '
        obj_weights = tf.placeholder(tf.float32)
        num_iter = tf.shape(obj_weights)[0]
        (data, labels) = dataset
        data = tf.constant(data)
        labels = tf.constant(labels)
        batches = tf.placeholder(tf.int32)
        first_unroll = tf.placeholder_with_default(False, [])
        reset_state = tf.placeholder_with_default(False, [])
        training_output = collections.namedtuple('TrainingOutput', ['metaobj', 'obj_weights', 'problem_objectives', 'initial_obj', 'batches', 'first_unroll', 'reset_state', 'output_state', 'init_loop_vars', 'output_loop_vars'])

        def loop_body(itr, obj_accum, params, attend_params, flattened_states, global_state, all_obj, unused_init_obj, data, labels, batches):
            if False:
                return 10
            "Body of the meta-training while loop for optimizing a sub-problem.\n\n      Args:\n        itr: The current meta-training iteration.\n        obj_accum: The accumulated objective over all training steps so far.\n        params: The parameters of the sub-problem.\n        attend_params: The parameters of the sub-problems at the attended\n            location.\n        flattened_states: The states of the trainable optimizer, sorted and\n            flattened into a list (since a while loop can't handle nested lists\n            or dictionaries).\n        global_state: The global state of the optimizer.\n        all_obj: The list of all objective values in the training process.\n        unused_init_obj: The initial objective (unused here, but needed in the\n            variable list because it's used in a stopping condition in the\n            loop_cond.)\n        data: The data for this problem.\n        labels: The labels corresponding to the data.\n        batches: The batch indexes needed for shuffled minibatch creation.\n\n      Returns:\n        itr: The updated meta-training iteration.\n        obj_accum: The updated accumulated objective.\n        params: The new parameters of the sub-problem.\n        attend_params: The new parameters of the sub-problems at the attended\n            location.\n        flattened_states: The new states of the trainable optimizer.\n        global_state: The updated global state.\n        all_obj: The updates list of all objective values.\n        unused_init_obj: The initial objective.\n        data: The data for this problem.\n        labels: The labels corresponding to the data.\n        batches: The batch indexes needed for shuffled minibatch creation.\n      "
            batch_indices = tf.gather(batches, itr)
            batch_data = tf.gather(data, batch_indices)
            batch_labels = tf.gather(labels, batch_indices)
            obj = problem.objective(params, data, labels)
            if self.use_attention:
                current_obj = problem.objective(attend_params, batch_data, batch_labels)
                grads = problem.gradients(current_obj, attend_params)
            else:
                current_obj = problem.objective(params, batch_data, batch_labels)
                grads = problem.gradients(current_obj, params)
            if not self.use_second_derivatives:
                new_grads = []
                for grad in grads:
                    if isinstance(grad, tf.IndexedSlices):
                        new_grads.append(tf.IndexedSlices(tf.stop_gradient(grad.values), grad.indices))
                    else:
                        new_grads.append(tf.stop_gradient(grad))
                grads = new_grads
            all_obj = tf.concat([all_obj, tf.reshape(obj, (1,))], 0)
            acc = tf.gather(obj_weights, itr) * obj
            obj_accum = tf.add(obj_accum, acc)
            obj_accum.set_shape([])
            dict_states = [dict(zip(self.state_keys, flat_state)) for flat_state in flattened_states]
            args = (params, grads, dict_states, global_state)
            updates = self._compute_updates(*args)
            (new_params, new_states, new_global_state, new_attend_params) = updates
            new_flattened_states = map(flatten_and_sort, new_states)
            return [itr + 1, obj_accum, new_params, new_attend_params, new_flattened_states, new_global_state, all_obj, unused_init_obj, data, labels, batches]

        def loop_cond(itr, obj_accum, unused_params, unused_attend_params, unused_flattened_states, unused_global_state, all_obj, init_obj, *args):
            if False:
                while True:
                    i = 10
            'Termination conditions of the sub-problem optimization loop.'
            del args
            cond1 = tf.less(itr, num_iter)
            cond2 = tf.is_finite(obj_accum)
            if self.obj_train_max_multiplier > 0:
                current_obj = tf.gather(all_obj, itr)
                max_diff = (self.obj_train_max_multiplier - 1) * tf.abs(init_obj)
                max_obj = init_obj + max_diff
                cond3 = tf.less(current_obj, max_obj)
                return tf.logical_and(tf.logical_and(cond1, cond2), cond3, name='training_loop_cond')
            else:
                return tf.logical_and(cond1, cond2, name='training_loop_cond')
        init = self._initialize_training_loop_parameters(problem, data, labels, batches, first_unroll, reset_state)
        (loop_vars, invariants, initial_obj, init_loop_vars_to_override) = init
        loop_output = tf.while_loop(loop_cond, loop_body, loop_vars, swap_memory=True, shape_invariants=invariants)
        (meta_obj, problem_objectives) = (loop_output[1], loop_output[6])
        scaled_meta_objective = self.scale_objective(meta_obj, problem_objectives, initial_obj)
        final_loop_vals = [initial_obj] + loop_output[2] + loop_output[3] + loop_output[5]
        final_loop_vals.extend(itertools.chain(*loop_output[4]))
        return training_output(scaled_meta_objective, obj_weights, problem_objectives, initial_obj, batches, first_unroll, reset_state, loop_output[4], init_loop_vars_to_override, final_loop_vals)

    def _initialize_training_loop_parameters(self, problem, data, labels, batches, first_unroll, reset_state):
        if False:
            print('Hello World!')
        'Initializes the vars and params needed for the training process.\n\n    Args:\n      problem: The problem being optimized.\n      data: The data for the problem.\n      labels: The corresponding labels for the data.\n      batches: The indexes needed to create shuffled batches of the data.\n      first_unroll: Whether this is the first unroll in a partial unrolling.\n      reset_state: Whether RNN state variables should be reset.\n\n    Returns:\n      loop_vars: The while loop variables for training.\n      invariants: The corresponding variable shapes (required by while loop).\n      initial_obj: The initial objective (used later for scaling).\n      init_loop_vars_to_override: The loop vars that can be overridden when\n          performing training via partial unrolls.\n    '
        initial_tensors = problem.init_tensors()
        return_initial_tensor_values = first_unroll
        (initial_params_vars, initial_params) = local_state_variables(initial_tensors, return_initial_tensor_values)
        (initial_attend_params_vars, initial_attend_params) = local_state_variables(initial_tensors, return_initial_tensor_values)
        initial_obj_init = problem.objective(initial_params, data, labels)
        return_initial_obj_init = first_unroll
        ([initial_obj_var], [initial_obj]) = local_state_variables([initial_obj_init], return_initial_obj_init)
        initial_itr = tf.constant(0, dtype=tf.int32)
        initial_meta_obj = tf.constant(0, dtype=tf.float32)
        initial_problem_objectives = tf.reshape(initial_obj_init, (1,))
        initial_state_vars = []
        initial_state = []
        state_shapes = []
        return_initial_state_values = reset_state
        for param in initial_tensors:
            (param_state_vars, param_state) = local_state_variables(flatten_and_sort(self._initialize_state(param)), return_initial_state_values)
            initial_state_vars.append(param_state_vars)
            initial_state.append(param_state)
            state_shapes.append([f.get_shape() for f in param_state])
        (initial_global_state_vars, initial_global_state) = local_state_variables(self._initialize_global_state(), return_initial_state_values)
        global_shapes = []
        for item in initial_global_state:
            global_shapes.append(item.get_shape())
        loop_vars = [initial_itr, initial_meta_obj, initial_params, initial_attend_params, initial_state, initial_global_state, initial_problem_objectives, initial_obj, data, labels, batches]
        invariants = [initial_itr.get_shape(), initial_meta_obj.get_shape(), [t.get_shape() for t in initial_params], [t.get_shape() for t in initial_attend_params], state_shapes, global_shapes, tensor_shape.TensorShape([None]), initial_obj.get_shape(), tensor_shape.unknown_shape(), tensor_shape.unknown_shape(), tensor_shape.unknown_shape()]
        init_loop_vars_to_override = [initial_obj_var] + initial_params_vars + initial_attend_params_vars + initial_global_state_vars
        init_loop_vars_to_override.extend(itertools.chain(*initial_state_vars))
        return (loop_vars, invariants, initial_obj, init_loop_vars_to_override)

    def scale_objective(self, total_obj, all_objs, initial_obj, obj_scale_eps=1e-06):
        if False:
            while True:
                i = 10
        'Normalizes the objective based on the initial objective value.\n\n    Args:\n      total_obj: The total accumulated objective over the training run.\n      all_objs: A list of all the individual objectives over the training run.\n      initial_obj: The initial objective value.\n      obj_scale_eps: The epsilon value to use in computations for stability.\n\n    Returns:\n      The scaled objective as a single value.\n    '
        if self.use_log_objective:
            if self.use_numerator_epsilon:
                scaled_problem_obj = (all_objs + obj_scale_eps) / (initial_obj + obj_scale_eps)
                log_scaled_problem_obj = tf.log(scaled_problem_obj)
            else:
                scaled_problem_obj = all_objs / (initial_obj + obj_scale_eps)
                log_scaled_problem_obj = tf.log(scaled_problem_obj + obj_scale_eps)
            return tf.reduce_mean(log_scaled_problem_obj)
        else:
            return total_obj / (initial_obj + obj_scale_eps)

def local_state_variables(init_values, return_init_values):
    if False:
        while True:
            i = 10
    "Create local variables initialized from init_values.\n\n  This will create local variables from a list of init_values. Each variable\n  will be named based on the value's shape and dtype.\n\n  As a convenience, a boolean tensor allows you to return value from\n  the created local variable or from the original init value.\n\n  Args:\n    init_values: iterable of tensors\n    return_init_values: boolean tensor\n\n  Returns:\n    local_vars: list of the created local variables.\n    vals: if return_init_values is true, then this returns the values of\n      init_values. Otherwise it returns the values of the local_vars.\n  "
    if not init_values:
        return ([], [])
    variable_use_count = tf.get_collection_ref(_LOCAL_STATE_VARIABLE_COLLECTION)
    if not variable_use_count:
        variable_use_count.append(collections.defaultdict(int))
    variable_use_count = variable_use_count[0]
    local_vars = []
    with tf.variable_scope(OPTIMIZER_SCOPE):
        for init_value in init_values:
            name = create_local_state_variable_name(init_value)
            unique_name = name + '_' + str(variable_use_count[name])
            variable_use_count[name] += 1
            local_vars.append(tf.get_local_variable(unique_name, initializer=tf.zeros(init_value.get_shape(), dtype=init_value.dtype)))
    vals = tf.cond(return_init_values, lambda : init_values, lambda : local_vars)
    if len(init_values) == 1:
        vals = [vals]
    return (local_vars, vals)

def create_local_state_variable_name(tensor):
    if False:
        for i in range(10):
            print('nop')
    'Create a name of the variable based on its type and shape.'
    if not tensor.get_shape().is_fully_defined():
        raise ValueError('Need a fully specified shape to create a local variable.')
    return _LOCAL_VARIABLE_PREFIX + '_'.join(map(str, tensor.get_shape().as_list())) + '_' + tensor.dtype.name

def is_local_state_variable(op):
    if False:
        i = 10
        return i + 15
    'Returns if this op is a local state variable created for training.'
    return op.node_def.op in ['Variable', 'VariableV2'] and op.name.startswith(OPTIMIZER_SCOPE + '/' + _LOCAL_VARIABLE_PREFIX)

def flatten_and_sort(dictionary):
    if False:
        for i in range(10):
            print('nop')
    'Flattens a dictionary into a list of values sorted by the keys.'
    return [dictionary[k] for k in sorted(dictionary.keys())]