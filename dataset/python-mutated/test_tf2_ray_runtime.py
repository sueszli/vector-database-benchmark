import pytest
from unittest import TestCase
import numpy as np
import tensorflow as tf
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2 import Estimator
NUM_TRAIN_SAMPLES = 1000
NUM_TEST_SAMPLES = 400

def linear_dataset(a=2, size=1000):
    if False:
        return 10
    x = np.random.rand(size)
    y = x / 2
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    return (x, y)

def create_train_datasets(config, batch_size):
    if False:
        return 10
    (x_train, y_train) = linear_dataset(size=NUM_TRAIN_SAMPLES)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(NUM_TRAIN_SAMPLES).batch(batch_size)
    return train_dataset

def create_test_dataset(config, batch_size):
    if False:
        while True:
            i = 10
    (x_test, y_test) = linear_dataset(size=NUM_TEST_SAMPLES)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    return test_dataset

def simple_model(config):
    if False:
        for i in range(10):
            print('nop')
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(10, input_shape=(1,)), tf.keras.layers.Dense(1)])
    return model

def compile_args(config):
    if False:
        return 10
    if 'lr' in config:
        lr = config['lr']
    else:
        lr = 0.001
    args = {'optimizer': tf.keras.optimizers.SGD(lr), 'loss': 'mean_squared_error', 'metrics': ['mean_squared_error']}
    return args

def model_creator(config):
    if False:
        i = 10
        return i + 15
    model = simple_model(config)
    model.compile(**compile_args(config))
    return model

class TestTF2Estimator(TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        init_orca_context(runtime='ray', address='localhost:6379')

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        stop_orca_context()

    def test_train(self):
        if False:
            return 10
        estimator = Estimator.from_keras(model_creator=model_creator, verbose=True, config=None, backend='ray', workers_per_node=2)
        start_stats = estimator.evaluate(create_test_dataset, batch_size=32, num_steps=2)
        print(start_stats)
        train_stats = estimator.fit(create_train_datasets, epochs=1, batch_size=32)
        print('This is Train Results:', train_stats)
        end_stats = estimator.evaluate(create_test_dataset, batch_size=32, num_steps=2)
        print('This is Val Results:', end_stats)
        assert isinstance(train_stats, dict), 'fit should return a dict'
        assert isinstance(end_stats, dict), 'evaluate should return a dict'
        assert estimator.get_model()
        dloss = end_stats['validation_loss'] - start_stats['validation_loss']
        dmse = end_stats['validation_mean_squared_error'] - start_stats['validation_mean_squared_error']
        print(f'dLoss: {dloss}, dMSE: {dmse}')
        assert dloss < 0 and dmse < 0, 'training sanity check failed. loss increased!'
if __name__ == '__main__':
    pytest.main([__file__])