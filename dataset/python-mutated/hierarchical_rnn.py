"""Collection of trainable optimizers for meta-optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import state_ops
from learned_optimizer.optimizer import rnn_cells
from learned_optimizer.optimizer import trainable_optimizer as opt
from learned_optimizer.optimizer import utils
tf.app.flags.DEFINE_float('biasgrucell_scale', 0.5, 'The scale for the internal BiasGRUCell vars.')
tf.app.flags.DEFINE_float('biasgrucell_gate_bias_init', 2.2, 'The bias for the internal BiasGRUCell reset and\n                             update gate variables.')
tf.app.flags.DEFINE_float('hrnn_rnn_readout_scale', 0.5, 'The initialization scale for the RNN readouts.')
tf.app.flags.DEFINE_float('hrnn_default_decay_var_init', 2.2, 'The default initializer value for any decay/\n                             momentum style variables and constants.\n                             sigmoid(2.2) ~ 0.9, sigmoid(-2.2) ~ 0.01.')
tf.app.flags.DEFINE_float('scale_decay_bias_init', 3.2, 'The initialization for the scale decay bias. This\n                             is the initial bias for the timescale for the\n                             exponential avg of the mean square gradients.')
tf.app.flags.DEFINE_float('learning_rate_momentum_logit_init', 3.2, 'Initialization for the learning rate momentum.')
tf.app.flags.DEFINE_float('hrnn_affine_scale', 0.5, 'The initialization scale for the weight matrix of\n                             the bias variables in layer0 and 1 of the hrnn.')
FLAGS = tf.flags.FLAGS

class HierarchicalRNN(opt.TrainableOptimizer):
    """3 level hierarchical RNN.

  Optionally uses second order gradient information and has decoupled evaluation
  and update locations.
  """

    def __init__(self, level_sizes, init_lr_range=(1e-06, 0.01), learnable_decay=True, dynamic_output_scale=True, use_attention=False, use_log_objective=True, num_gradient_scales=4, zero_init_lr_weights=True, use_log_means_squared=True, use_relative_lr=True, use_extreme_indicator=False, max_log_lr=33, obj_train_max_multiplier=-1, use_problem_lr_mean=False, use_gradient_shortcut=False, use_lr_shortcut=False, use_grad_products=False, use_multiple_scale_decays=False, learnable_inp_decay=True, learnable_rnn_init=True, random_seed=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Initializes the RNN per-parameter optimizer.\n\n    The hierarchy consists of up to three levels:\n    Level 0: per parameter RNN\n    Level 1: per tensor RNN\n    Level 2: global RNN\n\n    Args:\n      level_sizes: list or tuple with 1, 2, or 3 integers, the number of units\n          in each RNN in the hierarchy (level0, level1, level2).\n          length 1: only coordinatewise rnn's will be used\n          length 2: coordinatewise and tensor-level rnn's will be used\n          length 3: a single global-level rnn will be used in addition to\n             coordinatewise and tensor-level\n      init_lr_range: the range in which to initialize the learning rates\n      learnable_decay: whether to learn weights that dynamically modulate the\n          input scale via RMS style decay\n      dynamic_output_scale: whether to learn weights that dynamically modulate\n          the output scale\n      use_attention: whether to use attention to train the optimizer\n      use_log_objective: whether to train on the log of the objective\n      num_gradient_scales: the number of scales to use for gradient history\n      zero_init_lr_weights: whether to initialize the lr weights to zero\n      use_log_means_squared: whether to track the log of the means_squared,\n          used as a measure of signal vs. noise in gradient.\n      use_relative_lr: whether to use the relative learning rate as an\n          input during training (requires learnable_decay=True)\n      use_extreme_indicator: whether to use the extreme indicator for learning\n          rates as an input during training (requires learnable_decay=True)\n      max_log_lr: the maximum log learning rate allowed during train or test\n      obj_train_max_multiplier: max objective increase during a training run\n      use_problem_lr_mean: whether to use the mean over all learning rates in\n          the problem when calculating the relative learning rate as opposed to\n          the per-tensor mean\n      use_gradient_shortcut: Whether to add a learned affine projection of the\n          gradient to the update delta in addition to the gradient function\n          computed by the RNN\n      use_lr_shortcut: Whether to add as input the difference between the log lr\n          and the desired log lr (1e-3)\n      use_grad_products: Whether to use gradient products in the rnn input.\n          Only applicable if num_gradient_scales > 1\n      use_multiple_scale_decays: Whether to use multiple scales for the scale\n          decay, as with input decay\n      learnable_inp_decay: Whether to learn the input decay weights and bias.\n      learnable_rnn_init: Whether to learn the RNN state initialization.\n      random_seed: Random seed for random variable initializers. (Default: None)\n      **kwargs: args passed to TrainableOptimizer's constructor\n\n    Raises:\n      ValueError: If level_sizes is not a length 1, 2, or 3 list.\n      ValueError: If there are any non-integer sizes in level_sizes.\n      ValueError: If the init lr range is not of length 2.\n      ValueError: If the init lr range is not a valid range (min > max).\n    "
        if len(level_sizes) not in [1, 2, 3]:
            raise ValueError('HierarchicalRNN only supports 1, 2, or 3 levels in the hierarchy, but {} were requested.'.format(len(level_sizes)))
        if any((not isinstance(level, int) for level in level_sizes)):
            raise ValueError('Level sizes must be integer values, were {}'.format(level_sizes))
        if len(init_lr_range) != 2:
            raise ValueError('Initial LR range must be len 2, was {}'.format(len(init_lr_range)))
        if init_lr_range[0] > init_lr_range[1]:
            raise ValueError('Initial LR range min is greater than max.')
        self.learnable_decay = learnable_decay
        self.dynamic_output_scale = dynamic_output_scale
        self.use_attention = use_attention
        self.use_log_objective = use_log_objective
        self.num_gradient_scales = num_gradient_scales
        self.zero_init_lr_weights = zero_init_lr_weights
        self.use_log_means_squared = use_log_means_squared
        self.use_relative_lr = use_relative_lr
        self.use_extreme_indicator = use_extreme_indicator
        self.max_log_lr = max_log_lr
        self.use_problem_lr_mean = use_problem_lr_mean
        self.use_gradient_shortcut = use_gradient_shortcut
        self.use_lr_shortcut = use_lr_shortcut
        self.use_grad_products = use_grad_products
        self.use_multiple_scale_decays = use_multiple_scale_decays
        self.learnable_inp_decay = learnable_inp_decay
        self.learnable_rnn_init = learnable_rnn_init
        self.random_seed = random_seed
        self.num_layers = len(level_sizes)
        self.init_lr_range = init_lr_range
        self.reuse_vars = None
        self.reuse_global_state = None
        self.cells = []
        self.init_vectors = []
        with tf.variable_scope(opt.OPTIMIZER_SCOPE):
            self._initialize_rnn_cells(level_sizes)
            cell_size = level_sizes[0]
            scale_factor = FLAGS.hrnn_rnn_readout_scale / math.sqrt(cell_size)
            scaled_init = tf.random_normal_initializer(0.0, scale_factor, seed=self.random_seed)
            self.update_weights = tf.get_variable('update_weights', shape=(cell_size, 1), initializer=scaled_init)
            if self.use_attention:
                self.attention_weights = tf.get_variable('attention_weights', initializer=self.update_weights.initialized_value())
            self._initialize_scale_decay((cell_size, 1), scaled_init)
            self._initialize_input_decay((cell_size, 1), scaled_init)
            self._initialize_lr((cell_size, 1), scaled_init)
        state_keys = ['parameter', 'layer', 'scl_decay', 'inp_decay', 'true_param']
        if self.dynamic_output_scale:
            state_keys.append('log_learning_rate')
        for i in range(self.num_gradient_scales):
            state_keys.append('grad_accum{}'.format(i + 1))
            state_keys.append('ms{}'.format(i + 1))
        super(HierarchicalRNN, self).__init__('hRNN', state_keys, use_attention=use_attention, use_log_objective=use_log_objective, obj_train_max_multiplier=obj_train_max_multiplier, **kwargs)

    def _initialize_rnn_cells(self, level_sizes):
        if False:
            print('Hello World!')
        'Initializes the RNN cells to use in the hierarchical RNN.'
        for level in range(self.num_layers):
            scope = 'Level{}_RNN'.format(level)
            with tf.variable_scope(scope):
                hcell = rnn_cells.BiasGRUCell(level_sizes[level], scale=FLAGS.biasgrucell_scale, gate_bias_init=FLAGS.biasgrucell_gate_bias_init, random_seed=self.random_seed)
                self.cells.append(hcell)
                if self.learnable_rnn_init:
                    self.init_vectors.append(tf.Variable(tf.random_uniform([1, hcell.state_size], -1.0, 1.0, seed=self.random_seed), name='init_vector'))
                else:
                    self.init_vectors.append(tf.random_uniform([1, hcell.state_size], -1.0, 1.0, seed=self.random_seed))

    def _initialize_scale_decay(self, weights_tensor_shape, scaled_init):
        if False:
            i = 10
            return i + 15
        'Initializes the scale decay weights and bias variables or tensors.\n\n    Args:\n      weights_tensor_shape: The shape the weight tensor should take.\n      scaled_init: The scaled initialization for the weights tensor.\n    '
        if self.learnable_decay:
            self.scl_decay_weights = tf.get_variable('scl_decay_weights', shape=weights_tensor_shape, initializer=scaled_init)
            scl_decay_bias_init = tf.constant_initializer(FLAGS.scale_decay_bias_init)
            self.scl_decay_bias = tf.get_variable('scl_decay_bias', shape=(1,), initializer=scl_decay_bias_init)
        else:
            self.scl_decay_weights = tf.zeros_like(self.update_weights)
            self.scl_decay_bias = tf.log(0.93 / (1.0 - 0.93))

    def _initialize_input_decay(self, weights_tensor_shape, scaled_init):
        if False:
            for i in range(10):
                print('nop')
        'Initializes the input scale decay weights and bias variables or tensors.\n\n    Args:\n      weights_tensor_shape: The shape the weight tensor should take.\n      scaled_init: The scaled initialization for the weights tensor.\n    '
        if self.learnable_decay and self.num_gradient_scales > 1 and self.learnable_inp_decay:
            self.inp_decay_weights = tf.get_variable('inp_decay_weights', shape=weights_tensor_shape, initializer=scaled_init)
            inp_decay_bias_init = tf.constant_initializer(FLAGS.hrnn_default_decay_var_init)
            self.inp_decay_bias = tf.get_variable('inp_decay_bias', shape=(1,), initializer=inp_decay_bias_init)
        else:
            self.inp_decay_weights = tf.zeros_like(self.update_weights)
            self.inp_decay_bias = tf.log(0.89 / (1.0 - 0.89))

    def _initialize_lr(self, weights_tensor_shape, scaled_init):
        if False:
            return 10
        'Initializes the learning rate weights and bias variables or tensors.\n\n    Args:\n      weights_tensor_shape: The shape the weight tensor should take.\n      scaled_init: The scaled initialization for the weights tensor.\n    '
        if self.dynamic_output_scale:
            zero_init = tf.constant_initializer(0.0)
            wt_init = zero_init if self.zero_init_lr_weights else scaled_init
            self.lr_weights = tf.get_variable('learning_rate_weights', shape=weights_tensor_shape, initializer=wt_init)
            self.lr_bias = tf.get_variable('learning_rate_bias', shape=(1,), initializer=zero_init)
        else:
            self.lr_weights = tf.zeros_like(self.update_weights)
            self.lr_bias = tf.zeros([1, 1])

    def _initialize_state(self, var):
        if False:
            print('Hello World!')
        'Return a dictionary mapping names of state variables to their values.'
        var_vectorized = tf.reshape(var, [-1, 1])
        ndim = var_vectorized.get_shape().as_list()[0]
        state = {'parameter': tf.ones([ndim, 1]) * self.init_vectors[0], 'scl_decay': tf.zeros_like(var_vectorized), 'inp_decay': tf.zeros_like(var_vectorized), 'true_param': var}
        if self.num_layers > 1:
            state['layer'] = tf.ones([1, 1]) * self.init_vectors[1]
        if self.dynamic_output_scale:
            min_lr = self.init_lr_range[0]
            max_lr = self.init_lr_range[1]
            if min_lr == max_lr:
                log_init_lr = tf.log(min_lr * tf.ones_like(var_vectorized))
            else:
                actual_vals = tf.random_uniform(var_vectorized.get_shape().as_list(), np.log(min_lr) / 2.0, np.log(max_lr) / 2.0, seed=self.random_seed)
                offset = tf.random_uniform((), np.log(min_lr) / 2.0, np.log(max_lr) / 2.0, seed=self.random_seed)
                log_init_lr = actual_vals + offset
            clipped = tf.clip_by_value(log_init_lr, -33, self.max_log_lr)
            state['log_learning_rate'] = clipped
        for i in range(self.num_gradient_scales):
            state['grad_accum{}'.format(i + 1)] = tf.zeros_like(var_vectorized)
            state['ms{}'.format(i + 1)] = tf.zeros_like(var_vectorized)
        return state

    def _initialize_global_state(self):
        if False:
            i = 10
            return i + 15
        if self.num_layers < 3:
            return []
        rnn_global_init = tf.ones([1, 1]) * self.init_vectors[2]
        return [rnn_global_init]

    def _compute_updates(self, params, grads, states, global_state):
        if False:
            print('Hello World!')
        updated_params = []
        updated_attention = []
        updated_states = []
        with tf.variable_scope(opt.OPTIMIZER_SCOPE):
            mean_log_lr = self._compute_mean_log_lr(states)
            for (param, grad_unflat, state) in zip(params, grads, states):
                with tf.variable_scope('PerTensor', reuse=self.reuse_vars):
                    self.reuse_vars = True
                    grad = tf.reshape(grad_unflat, [-1, 1])
                    (grads_scaled, mean_squared_gradients, grads_accum) = self._compute_scaled_and_ms_grads(grad, state)
                    rnn_input = [g for g in grads_scaled]
                    self._extend_rnn_input(rnn_input, state, grads_scaled, mean_squared_gradients, mean_log_lr)
                    rnn_input_tensor = tf.concat(rnn_input, 1)
                    (layer_state, new_param_state) = self._update_rnn_cells(state, global_state, rnn_input_tensor, len(rnn_input) != len(grads_scaled))
                    (scl_decay, inp_decay, new_log_lr, update_step, lr_attend, attention_delta) = self._compute_rnn_state_projections(state, new_param_state, grads_scaled)
                    if self.use_attention:
                        truth = state['true_param']
                        updated_param = truth - update_step
                        attention_step = tf.reshape(lr_attend * attention_delta, truth.get_shape())
                        updated_attention.append(truth - attention_step)
                    else:
                        updated_param = param - update_step
                        updated_attention.append(updated_param)
                    updated_params.append(updated_param)
                    new_state = {'parameter': new_param_state, 'scl_decay': scl_decay, 'inp_decay': inp_decay, 'true_param': updated_param}
                    if layer_state is not None:
                        new_state['layer'] = layer_state
                    if self.dynamic_output_scale:
                        new_state['log_learning_rate'] = new_log_lr
                    for i in range(self.num_gradient_scales):
                        new_state['grad_accum{}'.format(i + 1)] = grads_accum[i]
                        new_state['ms{}'.format(i + 1)] = mean_squared_gradients[i]
                    updated_states.append(new_state)
            updated_global_state = self._compute_updated_global_state([layer_state], global_state)
        return (updated_params, updated_states, [updated_global_state], updated_attention)

    def _compute_mean_log_lr(self, states):
        if False:
            while True:
                i = 10
        'Computes the mean log learning rate across all variables.'
        if self.use_problem_lr_mean and self.use_relative_lr:
            sum_log_lr = 0.0
            count_log_lr = 0.0
            for state in states:
                sum_log_lr += tf.reduce_sum(state['log_learning_rate'])
                count_log_lr += state['log_learning_rate'].get_shape().num_elements()
            return sum_log_lr / count_log_lr

    def _compute_scaled_and_ms_grads(self, grad, state):
        if False:
            return 10
        'Computes the scaled gradient and the mean squared gradients.\n\n    Gradients are also accumulated across different timescales if appropriate.\n\n    Args:\n      grad: The gradient tensor for this layer.\n      state: The optimizer state for this layer.\n\n    Returns:\n      The scaled gradients, mean squared gradients, and accumulated gradients.\n    '
        input_decays = [state['inp_decay']]
        scale_decays = [state['scl_decay']]
        if self.use_multiple_scale_decays and self.num_gradient_scales > 1:
            for i in range(self.num_gradient_scales - 1):
                scale_decays.append(tf.sqrt(scale_decays[i]))
        for i in range(self.num_gradient_scales - 1):
            input_decays.append(tf.sqrt(input_decays[i]))
        grads_accum = []
        grads_scaled = []
        mean_squared_gradients = []
        if self.num_gradient_scales > 0:
            for (i, decay) in enumerate(input_decays):
                if self.num_gradient_scales == 1:
                    grad_accum = grad
                else:
                    old_accum = state['grad_accum{}'.format(i + 1)]
                    grad_accum = grad * (1.0 - decay) + old_accum * decay
                grads_accum.append(grad_accum)
                sd = scale_decays[i if self.use_multiple_scale_decays else 0]
                (grad_scaled, ms) = utils.rms_scaling(grad_accum, sd, state['ms{}'.format(i + 1)], update_ms=True)
                grads_scaled.append(grad_scaled)
                mean_squared_gradients.append(ms)
        return (grads_scaled, mean_squared_gradients, grads_accum)

    def _extend_rnn_input(self, rnn_input, state, grads_scaled, mean_squared_gradients, mean_log_lr):
        if False:
            i = 10
            return i + 15
        'Computes additional rnn inputs and adds them to the rnn_input list.'
        if self.num_gradient_scales > 1 and self.use_grad_products:
            grad_products = [a * b for (a, b) in zip(grads_scaled[:-1], grads_scaled[1:])]
            rnn_input.extend([g for g in grad_products])
        if self.use_log_means_squared:
            log_means_squared = [tf.log(ms + 1e-16) for ms in mean_squared_gradients]
            avg = tf.reduce_mean(log_means_squared, axis=0)
            mean_log_means_squared = [m - avg for m in log_means_squared]
            rnn_input.extend([m for m in mean_log_means_squared])
        if self.use_relative_lr or self.use_extreme_indicator:
            if not self.dynamic_output_scale:
                raise Exception('Relative LR and Extreme Indicator features require dynamic_output_scale to be set to True.')
            log_lr_vec = tf.reshape(state['log_learning_rate'], [-1, 1])
            if self.use_relative_lr:
                if self.use_problem_lr_mean:
                    relative_lr = log_lr_vec - mean_log_lr
                else:
                    relative_lr = log_lr_vec - tf.reduce_mean(log_lr_vec)
                rnn_input.append(relative_lr)
            if self.use_extreme_indicator:
                extreme_indicator = tf.nn.relu(log_lr_vec - tf.log(1.0)) - tf.nn.relu(tf.log(1e-06) - log_lr_vec)
                rnn_input.append(extreme_indicator)
        if self.use_lr_shortcut:
            log_lr_vec = tf.reshape(state['log_learning_rate'], [-1, 1])
            rnn_input.append(log_lr_vec - tf.log(0.001))

    def _update_rnn_cells(self, state, global_state, rnn_input_tensor, use_additional_features):
        if False:
            while True:
                i = 10
        'Updates the component RNN cells with the given state and tensor.\n\n    Args:\n      state: The current state of the optimizer.\n      global_state: The current global RNN state.\n      rnn_input_tensor: The input tensor to the RNN.\n      use_additional_features: Whether the rnn input tensor contains additional\n          features beyond the scaled gradients (affects whether the rnn input\n          tensor is used as input to the RNN.)\n\n    Returns:\n      layer_state: The new state of the per-tensor RNN.\n      new_param_state: The new state of the per-parameter RNN.\n    '
        with tf.variable_scope('Layer0_RNN'):
            total_bias = None
            if self.num_layers > 1:
                sz = 3 * self.cells[0].state_size
                param_bias = utils.affine([state['layer']], sz, scope='Param/Affine', scale=FLAGS.hrnn_affine_scale, random_seed=self.random_seed)
                total_bias = param_bias
                if self.num_layers == 3:
                    global_bias = utils.affine(global_state, sz, scope='Global/Affine', scale=FLAGS.hrnn_affine_scale, random_seed=self.random_seed)
                    total_bias += global_bias
            (new_param_state, _) = self.cells[0](rnn_input_tensor, state['parameter'], bias=total_bias)
        if self.num_layers > 1:
            with tf.variable_scope('Layer1_RNN'):
                if not use_additional_features:
                    layer_input = tf.reduce_mean(new_param_state, 0, keep_dims=True)
                else:
                    layer_input = tf.reduce_mean(tf.concat((new_param_state, rnn_input_tensor), 1), 0, keep_dims=True)
                if self.num_layers == 3:
                    sz = 3 * self.cells[1].state_size
                    layer_bias = utils.affine(global_state, sz, scale=FLAGS.hrnn_affine_scale, random_seed=self.random_seed)
                    (layer_state, _) = self.cells[1](layer_input, state['layer'], bias=layer_bias)
                else:
                    (layer_state, _) = self.cells[1](layer_input, state['layer'])
        else:
            layer_state = None
        return (layer_state, new_param_state)

    def _compute_rnn_state_projections(self, state, new_param_state, grads_scaled):
        if False:
            i = 10
            return i + 15
        'Computes the RNN state-based updates to parameters and update steps.'
        update_weights = self.update_weights
        update_delta = utils.project(new_param_state, update_weights)
        if self.use_gradient_shortcut:
            grads_scaled_tensor = tf.concat([g for g in grads_scaled], 1)
            update_delta += utils.affine(grads_scaled_tensor, 1, scope='GradsToDelta', include_bias=False, vec_mean=1.0 / len(grads_scaled), random_seed=self.random_seed)
        if self.dynamic_output_scale:
            denom = tf.sqrt(tf.reduce_mean(update_delta ** 2) + 1e-16)
            update_delta /= denom
        if self.use_attention:
            attention_weights = self.attention_weights
            attention_delta = utils.project(new_param_state, attention_weights)
            if self.use_gradient_shortcut:
                attention_delta += utils.affine(grads_scaled_tensor, 1, scope='GradsToAttnDelta', include_bias=False, vec_mean=1.0 / len(grads_scaled), random_seed=self.random_seed)
            if self.dynamic_output_scale:
                attention_delta /= tf.sqrt(tf.reduce_mean(attention_delta ** 2) + 1e-16)
        else:
            attention_delta = None
        scl_decay = utils.project(new_param_state, self.scl_decay_weights, bias=self.scl_decay_bias, activation=tf.nn.sigmoid)
        inp_decay = utils.project(new_param_state, self.inp_decay_weights, bias=self.inp_decay_bias, activation=tf.nn.sigmoid)
        (lr_param, lr_attend, new_log_lr) = self._compute_new_learning_rate(state, new_param_state)
        update_step = tf.reshape(lr_param * update_delta, state['true_param'].get_shape())
        return (scl_decay, inp_decay, new_log_lr, update_step, lr_attend, attention_delta)

    def _compute_new_learning_rate(self, state, new_param_state):
        if False:
            for i in range(10):
                print('nop')
        if self.dynamic_output_scale:
            lr_change = utils.project(new_param_state, self.lr_weights, bias=self.lr_bias)
            step_log_lr = state['log_learning_rate'] + lr_change
            step_log_lr += tf.stop_gradient(tf.clip_by_value(step_log_lr, -33, self.max_log_lr) - step_log_lr)
            lr_momentum_logit = tf.get_variable('learning_rate_momentum_logit', initializer=FLAGS.learning_rate_momentum_logit_init)
            lrm = tf.nn.sigmoid(lr_momentum_logit)
            new_log_lr = lrm * state['log_learning_rate'] + (1.0 - lrm) * step_log_lr
            param_stepsize_offset = tf.get_variable('param_stepsize_offset', initializer=-1.0)
            lr_param = tf.exp(step_log_lr + param_stepsize_offset)
            lr_attend = tf.exp(step_log_lr) if self.use_attention else lr_param
        else:
            lr_param = 2.0 * utils.project(new_param_state, self.lr_weights, bias=self.lr_bias, activation=tf.nn.sigmoid)
            new_log_lr = None
            lr_attend = lr_param
        return (lr_param, lr_attend, new_log_lr)

    def _compute_updated_global_state(self, layer_states, global_state):
        if False:
            for i in range(10):
                print('nop')
        'Computes the new global state gives the layers states and old state.\n\n    Args:\n      layer_states: The current layer states.\n      global_state: The old global state.\n\n    Returns:\n      The updated global state.\n    '
        updated_global_state = []
        if self.num_layers == 3:
            with tf.variable_scope('Layer2_RNN', reuse=self.reuse_global_state):
                self.reuse_global_state = True
                global_input = tf.reduce_mean(tf.concat(layer_states, 0), 0, keep_dims=True)
                (updated_global_state, _) = self.cells[2](global_input, global_state[0])
        return updated_global_state

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        if False:
            return 10
        'Overwrites the tf.train.Optimizer interface for applying gradients.'
        grads_and_vars = tuple(grads_and_vars)
        for (g, v) in grads_and_vars:
            if not isinstance(g, (tf.Tensor, tf.IndexedSlices, type(None))):
                raise TypeError('Gradient must be a Tensor, IndexedSlices, or None: %s' % g)
            if not isinstance(v, tf.Variable):
                raise TypeError('Variable must be a tf.Variable: %s' % v)
            if g is not None:
                self._assert_valid_dtypes([g, v])
        var_list = [v for (g, v) in grads_and_vars if g is not None]
        if not var_list:
            raise ValueError('No gradients provided for any variable: %s' % (grads_and_vars,))
        with tf.control_dependencies(None):
            self._create_slots(var_list)
        with tf.op_scope([], name, self._name) as name:
            with tf.variable_scope(self._name, reuse=self.reuse_global_state):
                gs = self._initialize_global_state()
                if gs:
                    global_state = [tf.get_variable('global_state', initializer=gs[0])]
                else:
                    global_state = []
            states = [{key: self.get_slot(var, key) for key in self.get_slot_names()} for var in var_list]
            (grads, params) = zip(*grads_and_vars)
            args = (params, grads, states, global_state)
            updates = self._compute_updates(*args)
            (new_params, new_states, new_global_state, new_attention) = updates
            update_ops = [tf.assign(gs, ngs) for (gs, ngs) in zip(global_state, new_global_state)]
            args = (params, states, new_params, new_attention, new_states)
            for (var, state, new_var, new_var_attend, new_state) in zip(*args):
                state_assign_ops = [tf.assign(state_var, new_state[key]) for (key, state_var) in state.items()]
                with tf.control_dependencies(state_assign_ops):
                    if self.use_attention:
                        param_update_op = var.assign(new_var_attend)
                    else:
                        param_update_op = var.assign(new_var)
                with tf.name_scope('update_' + var.op.name):
                    update_ops.append(param_update_op)
            real_params = [self.get_slot(var, 'true_param') for var in var_list]
            if global_step is None:
                return (self._finish(update_ops, name), real_params)
            else:
                with tf.control_dependencies([self._finish(update_ops, 'update')]):
                    return (state_ops.assign_add(global_step, 1, name=name).op, real_params)