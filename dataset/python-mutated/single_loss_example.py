"""A simple network to use in tests and examples."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import step_fn
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.layers import core
from tensorflow.python.layers import normalization
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

def single_loss_example(optimizer_fn, distribution, use_bias=False, iterations_per_step=1):
    if False:
        i = 10
        return i + 15
    'Build a very simple network to use in tests and examples.'

    def dataset_fn():
        if False:
            while True:
                i = 10
        return dataset_ops.Dataset.from_tensors([[1.0]]).repeat()
    optimizer = optimizer_fn()
    layer = core.Dense(1, use_bias=use_bias)

    def loss_fn(ctx, x):
        if False:
            for i in range(10):
                print('nop')
        del ctx
        y = array_ops.reshape(layer(x), []) - constant_op.constant(1.0)
        return y * y
    single_loss_step = step_fn.StandardSingleLossStep(dataset_fn, loss_fn, optimizer, distribution, iterations_per_step)
    return (single_loss_step, layer)

def minimize_loss_example(optimizer, use_bias=False, use_callable_loss=True):
    if False:
        print('Hello World!')
    'Example of non-distribution-aware legacy code.'

    def dataset_fn():
        if False:
            while True:
                i = 10
        dataset = dataset_ops.Dataset.from_tensors([[1.0]]).repeat()
        return dataset.batch(1, drop_remainder=True)
    layer = core.Dense(1, use_bias=use_bias)

    def model_fn(x):
        if False:
            i = 10
            return i + 15
        'A very simple model written by the user.'

        def loss_fn():
            if False:
                return 10
            y = array_ops.reshape(layer(x), []) - constant_op.constant(1.0)
            return y * y
        if strategy_test_lib.is_optimizer_v2_instance(optimizer):
            return optimizer.minimize(loss_fn, lambda : layer.trainable_variables)
        elif use_callable_loss:
            return optimizer.minimize(loss_fn)
        else:
            return optimizer.minimize(loss_fn())
    return (model_fn, dataset_fn, layer)

def batchnorm_example(optimizer_fn, batch_per_epoch=1, momentum=0.9, renorm=False, update_ops_in_replica_mode=False):
    if False:
        return 10
    'Example of non-distribution-aware legacy code with batch normalization.'

    def dataset_fn():
        if False:
            return 10
        return dataset_ops.Dataset.from_tensor_slices([[[float(x * 8 + y + z * 100) for y in range(8)] for x in range(16)] for z in range(batch_per_epoch)]).repeat()
    optimizer = optimizer_fn()
    batchnorm = normalization.BatchNormalization(renorm=renorm, momentum=momentum, fused=False)
    layer = core.Dense(1, use_bias=False)

    def model_fn(x):
        if False:
            return 10
        'A model that uses batchnorm.'

        def loss_fn():
            if False:
                return 10
            y = batchnorm(x, training=True)
            with ops.control_dependencies(ops.get_collection(ops.GraphKeys.UPDATE_OPS) if update_ops_in_replica_mode else []):
                loss = math_ops.reduce_mean(math_ops.reduce_sum(layer(y)) - constant_op.constant(1.0))
            return loss
        if strategy_test_lib.is_optimizer_v2_instance(optimizer):
            return optimizer.minimize(loss_fn, lambda : layer.trainable_variables)
        return optimizer.minimize(loss_fn)
    return (model_fn, dataset_fn, batchnorm)