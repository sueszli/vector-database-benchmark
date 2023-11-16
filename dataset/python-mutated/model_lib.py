"""Library with common functions for training and eval."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2

def default_hparams():
    if False:
        return 10
    'Returns default hyperparameters.'
    return tf.contrib.training.HParams(batch_size=32, eval_batch_size=50, weight_decay=0.0001, label_smoothing=0.1, train_adv_method='clean', train_lp_weight=0.0, optimizer='rms', momentum=0.9, rmsprop_decay=0.9, rmsprop_epsilon=1.0, lr_schedule='exp_decay', learning_rate=0.045, lr_decay_factor=0.94, lr_num_epochs_per_decay=2.0, lr_list=[1.0 / 6, 2.0 / 6, 3.0 / 6, 4.0 / 6, 5.0 / 6, 1.0, 0.1, 0.01, 0.001, 0.0001], lr_decay_epochs=[1, 2, 3, 4, 5, 30, 60, 80, 90])

def get_lr_schedule(hparams, examples_per_epoch, replicas_to_aggregate=1):
    if False:
        return 10
    'Returns TensorFlow op which compute learning rate.\n\n  Args:\n    hparams: hyper parameters.\n    examples_per_epoch: number of training examples per epoch.\n    replicas_to_aggregate: number of training replicas running in parallel.\n\n  Raises:\n    ValueError: if learning rate schedule specified in hparams is incorrect.\n\n  Returns:\n    learning_rate: tensor with learning rate.\n    steps_per_epoch: number of training steps per epoch.\n  '
    global_step = tf.train.get_or_create_global_step()
    steps_per_epoch = float(examples_per_epoch) / float(hparams.batch_size)
    if replicas_to_aggregate > 0:
        steps_per_epoch /= replicas_to_aggregate
    if hparams.lr_schedule == 'exp_decay':
        decay_steps = long(steps_per_epoch * hparams.lr_num_epochs_per_decay)
        learning_rate = tf.train.exponential_decay(hparams.learning_rate, global_step, decay_steps, hparams.lr_decay_factor, staircase=True)
    elif hparams.lr_schedule == 'step':
        lr_decay_steps = [long(epoch * steps_per_epoch) for epoch in hparams.lr_decay_epochs]
        learning_rate = tf.train.piecewise_constant(global_step, lr_decay_steps, hparams.lr_list)
    elif hparams.lr_schedule == 'fixed':
        learning_rate = hparams.learning_rate
    else:
        raise ValueError('Invalid value of lr_schedule: %s' % hparams.lr_schedule)
    if replicas_to_aggregate > 0:
        learning_rate *= replicas_to_aggregate
    return (learning_rate, steps_per_epoch)

def get_optimizer(hparams, learning_rate):
    if False:
        for i in range(10):
            print('nop')
    'Returns optimizer.\n\n  Args:\n    hparams: hyper parameters.\n    learning_rate: learning rate tensor.\n\n  Raises:\n    ValueError: if type of optimizer specified in hparams is incorrect.\n\n  Returns:\n    Instance of optimizer class.\n  '
    if hparams.optimizer == 'rms':
        optimizer = tf.train.RMSPropOptimizer(learning_rate, hparams.rmsprop_decay, hparams.momentum, hparams.rmsprop_epsilon)
    elif hparams.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, hparams.momentum)
    elif hparams.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    else:
        raise ValueError('Invalid value of optimizer: %s' % hparams.optimizer)
    return optimizer
RESNET_MODELS = {'resnet_v2_50': resnet_v2.resnet_v2_50}

def get_model(model_name, num_classes):
    if False:
        print('Hello World!')
    'Returns function which creates model.\n\n  Args:\n    model_name: Name of the model.\n    num_classes: Number of classes.\n\n  Raises:\n    ValueError: If model_name is invalid.\n\n  Returns:\n    Function, which creates model when called.\n  '
    if model_name.startswith('resnet'):

        def resnet_model(images, is_training, reuse=tf.AUTO_REUSE):
            if False:
                i = 10
                return i + 15
            with tf.contrib.framework.arg_scope(resnet_v2.resnet_arg_scope()):
                resnet_fn = RESNET_MODELS[model_name]
                (logits, _) = resnet_fn(images, num_classes, is_training=is_training, reuse=reuse)
                logits = tf.reshape(logits, [-1, num_classes])
            return logits
        return resnet_model
    else:
        raise ValueError('Invalid model: %s' % model_name)

def filter_trainable_variables(trainable_scopes):
    if False:
        while True:
            i = 10
    'Keep only trainable variables which are prefixed with given scopes.\n\n  Args:\n    trainable_scopes: either list of trainable scopes or string with comma\n      separated list of trainable scopes.\n\n  This function removes all variables which are not prefixed with given\n  trainable_scopes from collection of trainable variables.\n  Useful during network fine tuning, when you only need to train subset of\n  variables.\n  '
    if not trainable_scopes:
        return
    if isinstance(trainable_scopes, six.string_types):
        trainable_scopes = [scope.strip() for scope in trainable_scopes.split(',')]
    trainable_scopes = {scope for scope in trainable_scopes if scope}
    if not trainable_scopes:
        return
    trainable_collection = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
    non_trainable_vars = [v for v in trainable_collection if not any([v.op.name.startswith(s) for s in trainable_scopes])]
    for v in non_trainable_vars:
        trainable_collection.remove(v)