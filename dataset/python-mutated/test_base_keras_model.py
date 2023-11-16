from unittest import TestCase
from bigdl.orca.automl.model.base_keras_model import KerasBaseModel, KerasModelBuilder
import numpy as np
import tensorflow as tf
import pytest

def get_linear_data(a=2, b=5, size=None):
    if False:
        while True:
            i = 10
    x = np.arange(0, 10, 10 / size, dtype=np.float32)
    y = a * x + b
    return (x, y)

def get_dataset(size, config):
    if False:
        for i in range(10):
            print('nop')
    data = get_linear_data(size=size)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(config['batch_size'])
    return dataset

def get_train_val_data():
    if False:
        return 10
    data = get_linear_data(size=1000)
    validation_data = get_linear_data(size=400)
    return (data, validation_data)

def get_train_data_creator():
    if False:
        return 10

    def train_data_creator(config):
        if False:
            for i in range(10):
                print('nop')
        return get_dataset(size=1000, config=config)
    return train_data_creator

def get_val_data_creator():
    if False:
        for i in range(10):
            print('nop')

    def val_data_creator(config):
        if False:
            print('Hello World!')
        return get_dataset(size=400, config=config)
    return val_data_creator

def model_creator_keras(config):
    if False:
        while True:
            i = 10
    'Returns a tf.keras model'
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])
    model.compile(loss='mse', optimizer='sgd', metrics=['mse'])
    return model

def model_creator_multiple_metrics(config):
    if False:
        i = 10
        return i + 15
    'Returns a tf.keras model'
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])
    model.compile(loss='mse', optimizer='sgd', metrics=['mse', 'mae'])
    return model

class TestBaseKerasModel(TestCase):
    (data, validation_data) = get_train_val_data()

    def test_fit_evaluate(self):
        if False:
            return 10
        modelBuilder_keras = KerasModelBuilder(model_creator_keras)
        model = modelBuilder_keras.build(config={'lr': 0.01, 'batch_size': 32})
        val_result = model.fit_eval(data=self.data, validation_data=self.validation_data, metric='mse', epochs=20)
        assert val_result.get('mse')

    def test_fit_eval_creator(self):
        if False:
            while True:
                i = 10
        data_creator = get_train_data_creator()
        validation_data_creator = get_val_data_creator()
        modelBuilder_keras = KerasModelBuilder(model_creator_keras)
        model = modelBuilder_keras.build(config={'lr': 0.01, 'batch_size': 32})
        val_result = model.fit_eval(data=data_creator, validation_data=validation_data_creator, metric='mse', epochs=20)
        assert val_result.get('mse')

    def test_fit_eval_default_metric(self):
        if False:
            while True:
                i = 10
        modelBuilder_keras = KerasModelBuilder(model_creator_keras)
        model = modelBuilder_keras.build(config={'lr': 0.01, 'batch_size': 32})
        val_result = model.fit_eval(data=self.data, validation_data=self.validation_data, epochs=20)
        hist_metric_name = tf.keras.metrics.get('mse').__name__
        assert val_result.get(hist_metric_name)

    def test_multiple_metrics_default(self):
        if False:
            i = 10
            return i + 15
        modelBuilder_keras = KerasModelBuilder(model_creator_multiple_metrics)
        model = modelBuilder_keras.build(config={'lr': 0.01, 'batch_size': 32})
        with pytest.raises(RuntimeError):
            model.fit_eval(data=self.data, validation_data=self.validation_data, epochs=20)

    def test_uncompiled_model(self):
        if False:
            for i in range(10):
                print('nop')

        def model_creator(config):
            if False:
                while True:
                    i = 10
            'Returns a tf.keras model'
            model = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])
            return model
        modelBuilder_keras = KerasModelBuilder(model_creator)
        with pytest.raises(RuntimeError):
            model = modelBuilder_keras.build(config={'lr': 0.01, 'batch_size': 32})
            model.fit_eval(data=self.data, validation_data=self.validation_data, metric='mse', epochs=20)

    def test_unaligned_metric_value(self):
        if False:
            while True:
                i = 10
        modelBuilder_keras = KerasModelBuilder(model_creator_keras)
        model = modelBuilder_keras.build(config={'lr': 0.01, 'batch_size': 32})
        with pytest.raises(RuntimeError):
            model.fit_eval(data=self.data, validation_data=self.validation_data, metric='mae', epochs=20)
if __name__ == '__main__':
    pytest.main([__file__])