import pytest
from unittest import TestCase
import ray
from ray.data import Dataset
import tensorflow as tf
from bigdl.orca.learn.tf2 import Estimator
from bigdl.orca import init_orca_context, stop_orca_context

def train_data_creator(a=5, b=10, size=1000):
    if False:
        return 10

    def get_dataset(a, b, size) -> Dataset:
        if False:
            for i in range(10):
                print('nop')
        items = [i / size for i in range(size)]
        dataset = ray.data.from_items([{'x': x, 'y': a * x + b} for x in items])
        return dataset
    train_dataset = get_dataset(a, b, size)
    return train_dataset

def val_data_creator(a=5, b=10, size=100):
    if False:
        print('Hello World!')

    def get_dataset(a, b, size) -> Dataset:
        if False:
            i = 10
            return i + 15
        items = [i / size for i in range(size)]
        dataset = ray.data.from_items([{'x': x, 'y': a * x + b} for x in items])
        return dataset
    val_dataset = get_dataset(a, b, size)
    return val_dataset

def simple_model(config):
    if False:
        print('Hello World!')
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(1,)), tf.keras.layers.Dense(10), tf.keras.layers.Dense(1)])
    return model

def compile_args(config):
    if False:
        while True:
            i = 10
    if config is None:
        lr = 0.001
    else:
        lr = config['lr']
    args = {'optimizer': tf.keras.optimizers.SGD(lr), 'loss': 'mean_squared_error', 'metrics': ['mean_squared_error']}
    return args

def model_creator(config):
    if False:
        i = 10
        return i + 15
    model = simple_model(config)
    model.compile(**compile_args(config))
    return model

def get_estimator(workers_per_node=2, model_fn=model_creator):
    if False:
        return 10
    estimator = Estimator.from_keras(model_creator=model_fn, config={'lr': 0.001}, workers_per_node=workers_per_node, backend='ray')
    return estimator

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

    def test_train_and_evaluate(self):
        if False:
            while True:
                i = 10
        orca_estimator = get_estimator(workers_per_node=2)
        train_dataset = train_data_creator()
        validation_dataset = val_data_creator()
        data_config_args = {'output_signature': (tf.TensorSpec(shape=(None, 1), dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.float32))}
        start_eval_stats = orca_estimator.evaluate(data=validation_dataset, num_steps=2, batch_size=32, label_cols='x', feature_cols=['y'], data_config=data_config_args)
        train_stats = orca_estimator.fit(data=train_dataset, epochs=2, batch_size=32, label_cols='x', feature_cols=['y'], data_config=data_config_args)
        print(train_stats)
        end_eval_stats = orca_estimator.evaluate(data=validation_dataset, num_steps=2, batch_size=32, label_cols='x', feature_cols=['y'], data_config=data_config_args)
        assert isinstance(train_stats, dict), 'fit should return a dict'
        assert isinstance(end_eval_stats, dict), 'evaluate should return a dict'
        assert orca_estimator.get_model()
        dloss = end_eval_stats['validation_loss'] - start_eval_stats['validation_loss']
        dmse = end_eval_stats['validation_mean_squared_error'] - start_eval_stats['validation_mean_squared_error']
        print(f'dLoss: {dloss}, dMSE: {dmse}')
        assert dloss < 0 and dmse < 0, 'training sanity check failed. loss increased!'
if __name__ == '__main__':
    pytest.main([__file__])