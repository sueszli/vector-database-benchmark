"""Learning rate schedule."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import numpy as np
import tensorflow.compat.v2 as tf
from official.modeling.hyperparams import params_dict

class StepLearningRateWithLinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Class to generate learning rate tensor."""

    def __init__(self, params):
        if False:
            i = 10
            return i + 15
        'Creates the step learning rate tensor with linear warmup.'
        super(StepLearningRateWithLinearWarmup, self).__init__()
        assert isinstance(params, (dict, params_dict.ParamsDict))
        if isinstance(params, dict):
            params = params_dict.ParamsDict(params)
        self._params = params

    def __call__(self, global_step):
        if False:
            while True:
                i = 10
        warmup_lr = self._params.warmup_learning_rate
        warmup_steps = self._params.warmup_steps
        init_lr = self._params.init_learning_rate
        lr_levels = self._params.learning_rate_levels
        lr_steps = self._params.learning_rate_steps
        linear_warmup = warmup_lr + tf.cast(global_step, dtype=tf.float32) / warmup_steps * (init_lr - warmup_lr)
        learning_rate = tf.where(global_step < warmup_steps, linear_warmup, init_lr)
        for (next_learning_rate, start_step) in zip(lr_levels, lr_steps):
            learning_rate = tf.where(global_step >= start_step, next_learning_rate, learning_rate)
        return learning_rate

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        return {'_params': self._params.as_dict()}

class CosineLearningRateWithLinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Class to generate learning rate tensor."""

    def __init__(self, params):
        if False:
            i = 10
            return i + 15
        'Creates the consine learning rate tensor with linear warmup.'
        super(CosineLearningRateWithLinearWarmup, self).__init__()
        assert isinstance(params, (dict, params_dict.ParamsDict))
        if isinstance(params, dict):
            params = params_dict.ParamsDict(params)
        self._params = params

    def __call__(self, global_step):
        if False:
            i = 10
            return i + 15
        global_step = tf.cast(global_step, dtype=tf.float32)
        warmup_lr = self._params.warmup_learning_rate
        warmup_steps = self._params.warmup_steps
        init_lr = self._params.init_learning_rate
        total_steps = self._params.total_steps
        linear_warmup = warmup_lr + global_step / warmup_steps * (init_lr - warmup_lr)
        cosine_learning_rate = init_lr * (tf.cos(np.pi * (global_step - warmup_steps) / (total_steps - warmup_steps)) + 1.0) / 2.0
        learning_rate = tf.where(global_step < warmup_steps, linear_warmup, cosine_learning_rate)
        return learning_rate

    def get_config(self):
        if False:
            print('Hello World!')
        return {'_params': self._params.as_dict()}

def learning_rate_generator(params):
    if False:
        return 10
    'The learning rate function generator.'
    if params.type == 'step':
        return StepLearningRateWithLinearWarmup(params)
    elif params.type == 'cosine':
        return CosineLearningRateWithLinearWarmup(params)
    else:
        raise ValueError('Unsupported learning rate type: {}.'.format(params.type))