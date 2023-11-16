"""Functions and classes related to optimization (weight updates)."""
import re
from typing import Callable, List, Optional, Union
import tensorflow as tf
try:
    from tensorflow.keras.optimizers.legacy import Adam
except ImportError:
    from tensorflow.keras.optimizers import Adam

class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Applies a warmup schedule on a given learning rate decay schedule.

    Args:
        initial_learning_rate (`float`):
            The initial learning rate for the schedule after the warmup (so this will be the learning rate at the end
            of the warmup).
        decay_schedule_fn (`Callable`):
            The schedule function to apply after the warmup for the rest of training.
        warmup_steps (`int`):
            The number of steps for the warmup part of training.
        power (`float`, *optional*, defaults to 1.0):
            The power to use for the polynomial warmup (defaults is a linear warmup).
        name (`str`, *optional*):
            Optional name prefix for the returned tensors during the schedule.
    """

    def __init__(self, initial_learning_rate: float, decay_schedule_fn: Callable, warmup_steps: int, power: float=1.0, name: str=None):
        if False:
            print('Hello World!')
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        if False:
            i = 10
            return i + 15
        with tf.name_scope(self.name or 'WarmUp') as name:
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(global_step_float < warmup_steps_float, lambda : warmup_learning_rate, lambda : self.decay_schedule_fn(step - self.warmup_steps), name=name)

    def get_config(self):
        if False:
            while True:
                i = 10
        return {'initial_learning_rate': self.initial_learning_rate, 'decay_schedule_fn': self.decay_schedule_fn, 'warmup_steps': self.warmup_steps, 'power': self.power, 'name': self.name}

def create_optimizer(init_lr: float, num_train_steps: int, num_warmup_steps: int, min_lr_ratio: float=0.0, adam_beta1: float=0.9, adam_beta2: float=0.999, adam_epsilon: float=1e-08, adam_clipnorm: Optional[float]=None, adam_global_clipnorm: Optional[float]=None, weight_decay_rate: float=0.0, power: float=1.0, include_in_weight_decay: Optional[List[str]]=None):
    if False:
        print('Hello World!')
    '\n    Creates an optimizer with a learning rate schedule using a warmup phase followed by a linear decay.\n\n    Args:\n        init_lr (`float`):\n            The desired learning rate at the end of the warmup phase.\n        num_train_steps (`int`):\n            The total number of training steps.\n        num_warmup_steps (`int`):\n            The number of warmup steps.\n        min_lr_ratio (`float`, *optional*, defaults to 0):\n            The final learning rate at the end of the linear decay will be `init_lr * min_lr_ratio`.\n        adam_beta1 (`float`, *optional*, defaults to 0.9):\n            The beta1 to use in Adam.\n        adam_beta2 (`float`, *optional*, defaults to 0.999):\n            The beta2 to use in Adam.\n        adam_epsilon (`float`, *optional*, defaults to 1e-8):\n            The epsilon to use in Adam.\n        adam_clipnorm (`float`, *optional*, defaults to `None`):\n            If not `None`, clip the gradient norm for each weight tensor to this value.\n        adam_global_clipnorm (`float`, *optional*, defaults to `None`)\n            If not `None`, clip gradient norm to this value. When using this argument, the norm is computed over all\n            weight tensors, as if they were concatenated into a single vector.\n        weight_decay_rate (`float`, *optional*, defaults to 0):\n            The weight decay to use.\n        power (`float`, *optional*, defaults to 1.0):\n            The power to use for PolynomialDecay.\n        include_in_weight_decay (`List[str]`, *optional*):\n            List of the parameter names (or re patterns) to apply weight decay to. If none is passed, weight decay is\n            applied to all parameters except bias and layer norm parameters.\n    '
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=init_lr, decay_steps=num_train_steps - num_warmup_steps, end_learning_rate=init_lr * min_lr_ratio, power=power)
    if num_warmup_steps:
        lr_schedule = WarmUp(initial_learning_rate=init_lr, decay_schedule_fn=lr_schedule, warmup_steps=num_warmup_steps)
    if weight_decay_rate > 0.0:
        optimizer = AdamWeightDecay(learning_rate=lr_schedule, weight_decay_rate=weight_decay_rate, beta_1=adam_beta1, beta_2=adam_beta2, epsilon=adam_epsilon, clipnorm=adam_clipnorm, global_clipnorm=adam_global_clipnorm, exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'], include_in_weight_decay=include_in_weight_decay)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=adam_beta1, beta_2=adam_beta2, epsilon=adam_epsilon, clipnorm=adam_clipnorm, global_clipnorm=adam_global_clipnorm)
    return (optimizer, lr_schedule)

class AdamWeightDecay(Adam):
    """
    Adam enables L2 weight decay and clip_by_global_norm on gradients. Just adding the square of the weights to the
    loss function is *not* the correct way of using L2 regularization/weight decay with Adam, since that will interact
    with the m and v parameters in strange ways as shown in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Instead we want to decay the weights in a manner that doesn't interact with the m/v parameters. This is equivalent
    to adding the square of the weights to the loss with plain (non-momentum) SGD.

    Args:
        learning_rate (`Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]`, *optional*, defaults to 0.001):
            The learning rate to use or a schedule.
        beta_1 (`float`, *optional*, defaults to 0.9):
            The beta1 parameter in Adam, which is the exponential decay rate for the 1st momentum estimates.
        beta_2 (`float`, *optional*, defaults to 0.999):
            The beta2 parameter in Adam, which is the exponential decay rate for the 2nd momentum estimates.
        epsilon (`float`, *optional*, defaults to 1e-07):
            The epsilon parameter in Adam, which is a small constant for numerical stability.
        amsgrad (`bool`, *optional*, defaults to `False`):
            Whether to apply AMSGrad variant of this algorithm or not, see [On the Convergence of Adam and
            Beyond](https://arxiv.org/abs/1904.09237).
        weight_decay_rate (`float`, *optional*, defaults to 0.0):
            The weight decay to apply.
        include_in_weight_decay (`List[str]`, *optional*):
            List of the parameter names (or re patterns) to apply weight decay to. If none is passed, weight decay is
            applied to all parameters by default (unless they are in `exclude_from_weight_decay`).
        exclude_from_weight_decay (`List[str]`, *optional*):
            List of the parameter names (or re patterns) to exclude from applying weight decay to. If a
            `include_in_weight_decay` is passed, the names in it will supersede this list.
        name (`str`, *optional*, defaults to `"AdamWeightDecay"`):
            Optional name for the operations created when applying gradients.
        kwargs (`Dict[str, Any]`, *optional*):
            Keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
            norm; `clipvalue` is clip gradients by value, `decay` is included for backward compatibility to allow time
            inverse decay of learning rate. `lr` is included for backward compatibility, recommended to use
            `learning_rate` instead.
    """

    def __init__(self, learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]=0.001, beta_1: float=0.9, beta_2: float=0.999, epsilon: float=1e-07, amsgrad: bool=False, weight_decay_rate: float=0.0, include_in_weight_decay: Optional[List[str]]=None, exclude_from_weight_decay: Optional[List[str]]=None, name: str='AdamWeightDecay', **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)
        self.weight_decay_rate = weight_decay_rate
        self._include_in_weight_decay = include_in_weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay

    @classmethod
    def from_config(cls, config):
        if False:
            print('Hello World!')
        'Creates an optimizer from its config with WarmUp custom object.'
        custom_objects = {'WarmUp': WarmUp}
        return super(AdamWeightDecay, cls).from_config(config, custom_objects=custom_objects)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        if False:
            for i in range(10):
                print('nop')
        super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[var_device, var_dtype]['weight_decay_rate'] = tf.constant(self.weight_decay_rate, name='adam_weight_decay_rate')

    def _decay_weights_op(self, var, learning_rate, apply_state):
        if False:
            for i in range(10):
                print('nop')
        do_decay = self._do_use_weight_decay(var.name)
        if do_decay:
            return var.assign_sub(learning_rate * var * apply_state[var.device, var.dtype.base_dtype]['weight_decay_rate'], use_locking=self._use_locking)
        return tf.no_op()

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        (grads, tvars) = list(zip(*grads_and_vars))
        return super(AdamWeightDecay, self).apply_gradients(zip(grads, tvars), name=name, **kwargs)

    def _get_lr(self, var_device, var_dtype, apply_state):
        if False:
            return 10
        'Retrieves the learning rate with the given state.'
        if apply_state is None:
            return (self._decayed_lr_t[var_dtype], {})
        apply_state = apply_state or {}
        coefficients = apply_state.get((var_device, var_dtype))
        if coefficients is None:
            coefficients = self._fallback_apply_state(var_device, var_dtype)
            apply_state[var_device, var_dtype] = coefficients
        return (coefficients['lr_t'], {'apply_state': apply_state})

    def _resource_apply_dense(self, grad, var, apply_state=None):
        if False:
            print('Hello World!')
        (lr_t, kwargs) = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay, self)._resource_apply_dense(grad, var, **kwargs)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        if False:
            for i in range(10):
                print('nop')
        (lr_t, kwargs) = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay, self)._resource_apply_sparse(grad, var, indices, **kwargs)

    def get_config(self):
        if False:
            return 10
        config = super().get_config()
        config.update({'weight_decay_rate': self.weight_decay_rate})
        return config

    def _do_use_weight_decay(self, param_name):
        if False:
            print('Hello World!')
        'Whether to use L2 weight decay for `param_name`.'
        if self.weight_decay_rate == 0:
            return False
        if self._include_in_weight_decay:
            for r in self._include_in_weight_decay:
                if re.search(r, param_name) is not None:
                    return True
        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

class GradientAccumulator(object):
    """
    Gradient accumulation utility. When used with a distribution strategy, the accumulator should be called in a
    replica context. Gradients will be accumulated locally on each replica and without synchronization. Users should
    then call `.gradients`, scale the gradients if required, and pass the result to `apply_gradients`.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'Initializes the accumulator.'
        self._gradients = []
        self._accum_steps = None

    @property
    def step(self):
        if False:
            return 10
        'Number of accumulated steps.'
        if self._accum_steps is None:
            self._accum_steps = tf.Variable(tf.constant(0, dtype=tf.int64), trainable=False, synchronization=tf.VariableSynchronization.ON_READ, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        return self._accum_steps.value()

    @property
    def gradients(self):
        if False:
            while True:
                i = 10
        'The accumulated gradients on the current replica.'
        if not self._gradients:
            raise ValueError('The accumulator should be called first to initialize the gradients')
        return [gradient.value() if gradient is not None else gradient for gradient in self._gradients]

    def __call__(self, gradients):
        if False:
            while True:
                i = 10
        'Accumulates `gradients` on the current replica.'
        if not self._gradients:
            _ = self.step
            self._gradients.extend([tf.Variable(tf.zeros_like(gradient), trainable=False, synchronization=tf.VariableSynchronization.ON_READ, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA) if gradient is not None else gradient for gradient in gradients])
        if len(gradients) != len(self._gradients):
            raise ValueError(f'Expected {len(self._gradients)} gradients, but got {len(gradients)}')
        for (accum_gradient, gradient) in zip(self._gradients, gradients):
            if accum_gradient is not None and gradient is not None:
                accum_gradient.assign_add(gradient)
        self._accum_steps.assign_add(1)

    def reset(self):
        if False:
            print('Hello World!')
        'Resets the accumulated gradients on the current replica.'
        if not self._gradients:
            return
        self._accum_steps.assign(0)
        for gradient in self._gradients:
            if gradient is not None:
                gradient.assign(tf.zeros_like(gradient))