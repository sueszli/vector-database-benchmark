import os
from unittest import TestCase
import numpy as np
import pytest
import tensorflow as tf
from bigdl.orca.learn.tf2 import Estimator
from bigdl.orca.ray import OrcaRayContext
from bigdl.orca import OrcaContext
NUM_TRAIN_SAMPLES = 1000
NUM_TEST_SAMPLES = 400
resource_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), '../../../resources')

def linear_dataset(a=2, size=1000):
    if False:
        i = 10
        return i + 15
    x = np.random.rand(size)
    y = x / 2
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    return (x, y)

def create_train_datasets(config, batch_size):
    if False:
        while True:
            i = 10
    (x_train, y_train) = linear_dataset(size=NUM_TRAIN_SAMPLES)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(NUM_TRAIN_SAMPLES).batch(batch_size)
    return train_dataset

def create_test_dataset(config, batch_size):
    if False:
        for i in range(10):
            print('nop')
    (x_test, y_test) = linear_dataset(size=NUM_TEST_SAMPLES)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    return test_dataset

def simple_model(config):
    if False:
        while True:
            i = 10
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(10, input_shape=(1,)), tf.keras.layers.Dense(1)])
    return model

def compile_args(config):
    if False:
        i = 10
        return i + 15
    if 'lr' in config:
        lr = config['lr']
    else:
        lr = 0.001
    args = {'optimizer': tf.keras.optimizers.SGD(lr), 'loss': 'mean_squared_error', 'metrics': ['mean_squared_error']}
    return args

def model_creator(config):
    if False:
        return 10
    model = simple_model(config)
    model.compile(**compile_args(config))
    return model

def create_auto_shard_datasets(config, batch_size):
    if False:
        while True:
            i = 10
    data_path = os.path.join(resource_path, 'orca/learn/test_auto_shard/*.csv')
    dataset = tf.data.Dataset.list_files(data_path)
    dataset = dataset.interleave(lambda x: tf.data.TextLineDataset(x))
    dataset = dataset.map(lambda x: tf.strings.to_number(x))
    dataset = dataset.map(lambda x: (x, x))
    dataset = dataset.batch(batch_size)
    return dataset

def create_auto_shard_model(config):
    if False:
        for i in range(10):
            print('nop')
    model = tf.keras.models.Sequential([tf.keras.layers.Lambda(lambda x: tf.identity(x))])
    return model

def create_auto_shard_compile_args(config):
    if False:
        while True:
            i = 10

    def loss_func(y1, y2):
        if False:
            for i in range(10):
                print('nop')
        return tf.abs(y1[0] - y1[1]) + tf.abs(y2[0] - y2[1])
    args = {'optimizer': tf.keras.optimizers.SGD(lr=0.0), 'loss': loss_func}
    return args

def auto_shard_model_creator(config):
    if False:
        while True:
            i = 10
    model = create_auto_shard_model(config)
    model.compile(**create_auto_shard_compile_args(config))
    return model

class LRChecker(tf.keras.callbacks.Callback):

    def __init__(self, *args):
        if False:
            return 10
        super(LRChecker, self).__init__(*args)
        self.warmup_lr = [0.16, 0.22, 0.28, 0.34, 0.4]

    def on_epoch_end(self, epoch, logs=None):
        if False:
            while True:
                i = 10
        current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        print('epoch {} current lr is {}'.format(epoch, current_lr))
        if epoch < 5:
            assert abs(current_lr - self.warmup_lr[epoch]) < 1e-05
        elif 5 <= epoch < 10:
            assert abs(current_lr - 0.4) < 1e-05
        elif 10 <= epoch < 15:
            assert abs(current_lr - 0.04) < 1e-05
        elif 15 <= epoch < 20:
            assert abs(current_lr - 0.004) < 1e-05
        else:
            assert abs(current_lr - 0.0004) < 1e-05

class TestTFHorovodEstimator(TestCase):

    def test_fit_and_evaluate_tf(self):
        if False:
            for i in range(10):
                print('nop')
        ray_ctx = OrcaRayContext.get()
        batch_size = 32
        global_batch_size = batch_size * ray_ctx.num_ray_nodes
        trainer = Estimator.from_keras(model_creator=simple_model, compile_args_creator=compile_args, verbose=True, config=None, backend='horovod')
        start_stats = trainer.evaluate(create_test_dataset, batch_size=global_batch_size, num_steps=NUM_TEST_SAMPLES // global_batch_size)
        print(start_stats)

        def scheduler(epoch):
            if False:
                i = 10
                return i + 15
            if epoch < 2:
                return 0.001
            else:
                return 0.001 * tf.math.exp(0.1 * (2 - epoch))
        scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
        trainer.fit(create_train_datasets, epochs=2, batch_size=global_batch_size, steps_per_epoch=10, callbacks=[scheduler])
        trainer.fit(create_train_datasets, epochs=2, batch_size=global_batch_size, steps_per_epoch=10, callbacks=[scheduler])
        end_stats = trainer.evaluate(create_test_dataset, batch_size=global_batch_size, num_steps=NUM_TEST_SAMPLES // global_batch_size)
        print(end_stats)
        dloss = end_stats['validation_loss'] - start_stats['validation_loss']
        dmse = end_stats['validation_mean_squared_error'] - start_stats['validation_mean_squared_error']
        print(f'dLoss: {dloss}, dMSE: {dmse}')
        assert dloss < 0 and dmse < 0, 'training sanity check failed. loss increased!'

    def test_auto_shard_horovod(self):
        if False:
            print('Hello World!')
        ray_ctx = OrcaRayContext.get()
        trainer = Estimator.from_keras(model_creator=create_auto_shard_model, compile_args_creator=create_auto_shard_compile_args, verbose=True, backend='horovod', workers_per_node=2)
        stats = trainer.fit(create_auto_shard_datasets, epochs=1, batch_size=4, steps_per_epoch=2)
        assert stats['loss'] == [0.0]

    def test_horovod_learning_rate_schedule(self):
        if False:
            for i in range(10):
                print('nop')
        import horovod
        (major, minor, patch) = horovod.__version__.split('.')
        larger_major = int(major) > 0
        larger_minor = int(major) == 0 and int(minor) > 19
        larger_patch = int(major) == 0 and int(minor) == 19 and (int(patch) >= 2)
        if larger_major or larger_minor or larger_patch:
            ray_ctx = OrcaRayContext.get()
            batch_size = 32
            workers_per_node = 4
            global_batch_size = batch_size * workers_per_node
            config = {'lr': 0.8}
            trainer = Estimator.from_keras(model_creator=simple_model, compile_args_creator=compile_args, verbose=True, config=config, backend='horovod', workers_per_node=workers_per_node)
            import horovod.tensorflow.keras as hvd
            callbacks = [hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, initial_lr=0.4, verbose=True), hvd.callbacks.LearningRateScheduleCallback(start_epoch=5, end_epoch=10, multiplier=1.0, initial_lr=0.4), hvd.callbacks.LearningRateScheduleCallback(start_epoch=10, end_epoch=15, multiplier=0.1, initial_lr=0.4), hvd.callbacks.LearningRateScheduleCallback(start_epoch=15, end_epoch=20, multiplier=0.01, initial_lr=0.4), hvd.callbacks.LearningRateScheduleCallback(start_epoch=20, multiplier=0.001, initial_lr=0.4), LRChecker()]
            for i in range(30):
                trainer.fit(create_train_datasets, epochs=1, batch_size=global_batch_size, callbacks=callbacks)
        else:
            pass
if __name__ == '__main__':
    pytest.main([__file__])