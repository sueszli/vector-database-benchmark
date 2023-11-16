import os
import numpy as np
import pytest
import keras
from keras import layers
from keras import models
from keras import ops
from keras import testing
from keras.legacy.saving import legacy_h5_format
from keras.saving import object_registration
from keras.saving import serialization_lib
tf_keras = None

def get_sequential_model(keras):
    if False:
        print('Hello World!')
    return keras.Sequential([keras.layers.Input((3,), batch_size=2), keras.layers.Dense(4, activation='relu'), keras.layers.BatchNormalization(moving_mean_initializer='uniform', gamma_initializer='uniform'), keras.layers.Dense(5, activation='softmax')])

def get_functional_model(keras):
    if False:
        for i in range(10):
            print('nop')
    inputs = keras.Input((3,), batch_size=2)
    x = keras.layers.Dense(4, activation='relu')(inputs)
    residual = x
    x = keras.layers.BatchNormalization(moving_mean_initializer='uniform', gamma_initializer='uniform')(x)
    x = keras.layers.Dense(4, activation='relu')(x)
    x = keras.layers.add([x, residual])
    outputs = keras.layers.Dense(5, activation='softmax')(x)
    return keras.Model(inputs, outputs)

def get_subclassed_model(keras):
    if False:
        i = 10
        return i + 15

    class MyModel(keras.Model):

        def __init__(self, **kwargs):
            if False:
                i = 10
                return i + 15
            super().__init__(**kwargs)
            self.dense_1 = keras.layers.Dense(3, activation='relu')
            self.dense_2 = keras.layers.Dense(1, activation='sigmoid')

        def call(self, x):
            if False:
                for i in range(10):
                    print('nop')
            return self.dense_2(self.dense_1(x))
    model = MyModel()
    model(np.random.random((2, 3)))
    return model

@pytest.mark.requires_trainable_backend
class LegacyH5WeightsTest(testing.TestCase):

    def _check_reloading_weights(self, ref_input, model, tf_keras_model):
        if False:
            return 10
        ref_output = tf_keras_model(ref_input)
        initial_weights = model.get_weights()
        temp_filepath = os.path.join(self.get_temp_dir(), 'weights.h5')
        tf_keras_model.save_weights(temp_filepath)
        model.load_weights(temp_filepath)
        output = model(ref_input)
        self.assertAllClose(ref_output, output, atol=1e-05)
        model.set_weights(initial_weights)
        model.load_weights(temp_filepath)
        output = model(ref_input)
        self.assertAllClose(ref_output, output, atol=1e-05)

    def DISABLED_test_sequential_model_weights(self):
        if False:
            return 10
        model = get_sequential_model(keras)
        tf_keras_model = get_sequential_model(tf_keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_weights(ref_input, model, tf_keras_model)

    def DISABLED_test_functional_model_weights(self):
        if False:
            print('Hello World!')
        model = get_functional_model(keras)
        tf_keras_model = get_functional_model(tf_keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_weights(ref_input, model, tf_keras_model)

    def DISABLED_test_subclassed_model_weights(self):
        if False:
            for i in range(10):
                print('nop')
        model = get_subclassed_model(keras)
        tf_keras_model = get_subclassed_model(tf_keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_weights(ref_input, model, tf_keras_model)

@pytest.mark.requires_trainable_backend
class LegacyH5WholeModelTest(testing.TestCase):

    def _check_reloading_model(self, ref_input, model):
        if False:
            i = 10
            return i + 15
        ref_output = model(ref_input)
        temp_filepath = os.path.join(self.get_temp_dir(), 'model.h5')
        legacy_h5_format.save_model_to_hdf5(model, temp_filepath)
        loaded = legacy_h5_format.load_model_from_hdf5(temp_filepath)
        output = loaded(ref_input)
        self.assertAllClose(ref_output, output, atol=1e-05)

    def DISABLED_test_sequential_model(self):
        if False:
            while True:
                i = 10
        model = get_sequential_model(keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_model(ref_input, model)

    def DISABLED_test_functional_model(self):
        if False:
            for i in range(10):
                print('nop')
        model = get_functional_model(keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_model(ref_input, model)

    def DISABLED_test_compiled_model_with_various_layers(self):
        if False:
            print('Hello World!')
        model = models.Sequential()
        model.add(layers.Dense(2, input_shape=(3,)))
        model.add(layers.RepeatVector(3))
        model.add(layers.TimeDistributed(layers.Dense(3)))
        model.compile(optimizer='rmsprop', loss='mse')
        ref_input = np.random.random((1, 3))
        self._check_reloading_model(ref_input, model)

    def DISABLED_test_saving_lambda(self):
        if False:
            while True:
                i = 10
        mean = ops.random.uniform((4, 2, 3))
        std = ops.abs(ops.random.uniform((4, 2, 3))) + 1e-05
        inputs = layers.Input(shape=(4, 2, 3))
        output = layers.Lambda(lambda image, mu, std: (image - mu) / std, arguments={'mu': mean, 'std': std})(inputs)
        model = models.Model(inputs, output)
        model.compile(loss='mse', optimizer='sgd', metrics=['acc'])
        temp_filepath = os.path.join(self.get_temp_dir(), 'lambda_model.h5')
        legacy_h5_format.save_model_to_hdf5(model, temp_filepath)
        loaded = legacy_h5_format.load_model_from_hdf5(temp_filepath)
        self.assertAllClose(mean, loaded.layers[1].arguments['mu'])
        self.assertAllClose(std, loaded.layers[1].arguments['std'])

    def DISABLED_test_saving_include_optimizer_false(self):
        if False:
            i = 10
            return i + 15
        model = models.Sequential()
        model.add(layers.Dense(1))
        model.compile('adam', loss='mse')
        (x, y) = (np.ones((10, 10)), np.ones((10, 1)))
        model.fit(x, y)
        ref_output = model(x)
        temp_filepath = os.path.join(self.get_temp_dir(), 'model.h5')
        legacy_h5_format.save_model_to_hdf5(model, temp_filepath, include_optimizer=False)
        loaded = legacy_h5_format.load_model_from_hdf5(temp_filepath)
        output = loaded(x)
        with self.assertRaises(AttributeError):
            _ = loaded.optimizer
        self.assertAllClose(ref_output, output, atol=1e-05)

    def DISABLED_test_custom_sequential_registered_no_scope(self):
        if False:
            while True:
                i = 10

        @object_registration.register_keras_serializable(package='my_package')
        class MyDense(layers.Dense):

            def __init__(self, units, **kwargs):
                if False:
                    print('Hello World!')
                super().__init__(units, **kwargs)
        inputs = layers.Input(shape=[1])
        custom_layer = MyDense(1)
        model = models.Sequential(layers=[inputs, custom_layer])
        ref_input = np.array([5])
        self._check_reloading_model(ref_input, model)

    def DISABLED_test_custom_functional_registered_no_scope(self):
        if False:
            i = 10
            return i + 15

        @object_registration.register_keras_serializable(package='my_package')
        class MyDense(layers.Dense):

            def __init__(self, units, **kwargs):
                if False:
                    while True:
                        i = 10
                super().__init__(units, **kwargs)
        inputs = layers.Input(shape=[1])
        outputs = MyDense(1)(inputs)
        model = models.Model(inputs, outputs)
        ref_input = np.array([5])
        self._check_reloading_model(ref_input, model)

    def DISABLED_test_nested_layers(self):
        if False:
            print('Hello World!')

        class MyLayer(layers.Layer):

            def __init__(self, sublayers, **kwargs):
                if False:
                    print('Hello World!')
                super().__init__(**kwargs)
                self.sublayers = sublayers

            def call(self, x):
                if False:
                    return 10
                prev_input = x
                for layer in self.sublayers:
                    prev_input = layer(prev_input)
                return prev_input

            def get_config(self):
                if False:
                    for i in range(10):
                        print('nop')
                config = super().get_config()
                config['sublayers'] = serialization_lib.serialize_keras_object(self.sublayers)
                return config

            @classmethod
            def from_config(cls, config):
                if False:
                    print('Hello World!')
                config['sublayers'] = serialization_lib.deserialize_keras_object(config['sublayers'])
                return cls(**config)

        @object_registration.register_keras_serializable(package='Foo')
        class RegisteredSubLayer(layers.Layer):
            pass
        layer = MyLayer([layers.Dense(2, name='MyDense'), RegisteredSubLayer(name='MySubLayer')])
        model = models.Sequential([layer])
        with self.subTest('test_JSON'):
            from keras.models.model import model_from_json
            model_json = model.to_json()
            self.assertIn('Foo>RegisteredSubLayer', model_json)
            loaded_model = model_from_json(model_json, custom_objects={'MyLayer': MyLayer})
            loaded_layer = loaded_model.layers[0]
            self.assertIsInstance(loaded_layer.sublayers[0], layers.Dense)
            self.assertEqual(loaded_layer.sublayers[0].name, 'MyDense')
            self.assertIsInstance(loaded_layer.sublayers[1], RegisteredSubLayer)
            self.assertEqual(loaded_layer.sublayers[1].name, 'MySubLayer')
        with self.subTest('test_H5'):
            temp_filepath = os.path.join(self.get_temp_dir(), 'model.h5')
            legacy_h5_format.save_model_to_hdf5(model, temp_filepath)
            loaded_model = legacy_h5_format.load_model_from_hdf5(temp_filepath, custom_objects={'MyLayer': MyLayer})
            loaded_layer = loaded_model.layers[0]
            self.assertIsInstance(loaded_layer.sublayers[0], layers.Dense)
            self.assertEqual(loaded_layer.sublayers[0].name, 'MyDense')
            self.assertIsInstance(loaded_layer.sublayers[1], RegisteredSubLayer)
            self.assertEqual(loaded_layer.sublayers[1].name, 'MySubLayer')

@pytest.mark.requires_trainable_backend
class LegacyH5BackwardsCompatTest(testing.TestCase):

    def _check_reloading_model(self, ref_input, model, tf_keras_model):
        if False:
            while True:
                i = 10
        ref_output = tf_keras_model(ref_input)
        temp_filepath = os.path.join(self.get_temp_dir(), 'model.h5')
        tf_keras_model.save(temp_filepath)
        loaded = legacy_h5_format.load_model_from_hdf5(temp_filepath)
        output = loaded(ref_input)
        self.assertAllClose(ref_output, output, atol=1e-05)

    def DISABLED_test_sequential_model(self):
        if False:
            print('Hello World!')
        model = get_sequential_model(keras)
        tf_keras_model = get_sequential_model(tf_keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_model(ref_input, model, tf_keras_model)

    def DISABLED_test_functional_model(self):
        if False:
            while True:
                i = 10
        tf_keras_model = get_functional_model(tf_keras)
        model = get_functional_model(keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_model(ref_input, model, tf_keras_model)

    def DISABLED_test_compiled_model_with_various_layers(self):
        if False:
            i = 10
            return i + 15
        model = models.Sequential()
        model.add(layers.Dense(2, input_shape=(3,)))
        model.add(layers.RepeatVector(3))
        model.add(layers.TimeDistributed(layers.Dense(3)))
        model.compile(optimizer='rmsprop', loss='mse')
        tf_keras_model = tf_keras.Sequential()
        tf_keras_model.add(tf_keras.layers.Dense(2, input_shape=(3,)))
        tf_keras_model.add(tf_keras.layers.RepeatVector(3))
        tf_keras_model.add(tf_keras.layers.TimeDistributed(tf_keras.layers.Dense(3)))
        tf_keras_model.compile(optimizer='rmsprop', loss='mse')
        ref_input = np.random.random((1, 3))
        self._check_reloading_model(ref_input, model, tf_keras_model)

    def DISABLED_test_saving_lambda(self):
        if False:
            return 10
        mean = np.random.random((4, 2, 3))
        std = np.abs(np.random.random((4, 2, 3))) + 1e-05
        inputs = tf_keras.layers.Input(shape=(4, 2, 3))
        output = tf_keras.layers.Lambda(lambda image, mu, std: (image - mu) / std, arguments={'mu': mean, 'std': std}, output_shape=inputs.shape)(inputs)
        tf_keras_model = tf_keras.Model(inputs, output)
        tf_keras_model.compile(loss='mse', optimizer='sgd', metrics=['acc'])
        temp_filepath = os.path.join(self.get_temp_dir(), 'lambda_model.h5')
        tf_keras_model.save(temp_filepath)
        loaded = legacy_h5_format.load_model_from_hdf5(temp_filepath)
        self.assertAllClose(mean, loaded.layers[1].arguments['mu'])
        self.assertAllClose(std, loaded.layers[1].arguments['std'])

    def DISABLED_test_saving_include_optimizer_false(self):
        if False:
            print('Hello World!')
        tf_keras_model = tf_keras.Sequential()
        tf_keras_model.add(tf_keras.layers.Dense(1))
        tf_keras_model.compile('adam', loss='mse')
        (x, y) = (np.ones((10, 10)), np.ones((10, 1)))
        tf_keras_model.fit(x, y)
        ref_output = tf_keras_model(x)
        temp_filepath = os.path.join(self.get_temp_dir(), 'model.h5')
        tf_keras_model.save(temp_filepath, include_optimizer=False)
        loaded = legacy_h5_format.load_model_from_hdf5(temp_filepath)
        output = loaded(x)
        with self.assertRaises(AttributeError):
            _ = loaded.optimizer
        self.assertAllClose(ref_output, output, atol=1e-05)

    def DISABLED_test_custom_sequential_registered_no_scope(self):
        if False:
            return 10

        @tf_keras.saving.register_keras_serializable(package='my_package')
        class MyDense(tf_keras.layers.Dense):

            def __init__(self, units, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__(units, **kwargs)
        inputs = tf_keras.layers.Input(shape=[1])
        custom_layer = MyDense(1)
        tf_keras_model = tf_keras.Sequential(layers=[inputs, custom_layer])

        @object_registration.register_keras_serializable(package='my_package')
        class MyDense(layers.Dense):

            def __init__(self, units, **kwargs):
                if False:
                    i = 10
                    return i + 15
                super().__init__(units, **kwargs)
        inputs = layers.Input(shape=[1])
        custom_layer = MyDense(1)
        model = models.Sequential(layers=[inputs, custom_layer])
        ref_input = np.array([5])
        self._check_reloading_model(ref_input, model, tf_keras_model)

    def DISABLED_test_custom_functional_registered_no_scope(self):
        if False:
            while True:
                i = 10

        @tf_keras.saving.register_keras_serializable(package='my_package')
        class MyDense(tf_keras.layers.Dense):

            def __init__(self, units, **kwargs):
                if False:
                    return 10
                super().__init__(units, **kwargs)
        inputs = tf_keras.layers.Input(shape=[1])
        outputs = MyDense(1)(inputs)
        tf_keras_model = tf_keras.Model(inputs, outputs)

        @object_registration.register_keras_serializable(package='my_package')
        class MyDense(layers.Dense):

            def __init__(self, units, **kwargs):
                if False:
                    print('Hello World!')
                super().__init__(units, **kwargs)
        inputs = layers.Input(shape=[1])
        outputs = MyDense(1)(inputs)
        model = models.Model(inputs, outputs)
        ref_input = np.array([5])
        self._check_reloading_model(ref_input, model, tf_keras_model)

    def DISABLED_test_nested_layers(self):
        if False:
            while True:
                i = 10

        class MyLayer(tf_keras.layers.Layer):

            def __init__(self, sublayers, **kwargs):
                if False:
                    i = 10
                    return i + 15
                super().__init__(**kwargs)
                self.sublayers = sublayers

            def call(self, x):
                if False:
                    return 10
                prev_input = x
                for layer in self.sublayers:
                    prev_input = layer(prev_input)
                return prev_input

            def get_config(self):
                if False:
                    for i in range(10):
                        print('nop')
                config = super().get_config()
                config['sublayers'] = tf_keras.saving.serialize_keras_object(self.sublayers)
                return config

            @classmethod
            def from_config(cls, config):
                if False:
                    i = 10
                    return i + 15
                config['sublayers'] = tf_keras.saving.deserialize_keras_object(config['sublayers'])
                return cls(**config)

        @tf_keras.saving.register_keras_serializable(package='Foo')
        class RegisteredSubLayer(layers.Layer):

            def call(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x
        layer = MyLayer([tf_keras.layers.Dense(2, name='MyDense'), RegisteredSubLayer(name='MySubLayer')])
        tf_keras_model = tf_keras.Sequential([layer])
        x = np.random.random((4, 2))
        ref_output = tf_keras_model(x)
        temp_filepath = os.path.join(self.get_temp_dir(), 'model.h5')
        tf_keras_model.save(temp_filepath)

        class MyLayer(layers.Layer):

            def __init__(self, sublayers, **kwargs):
                if False:
                    i = 10
                    return i + 15
                super().__init__(**kwargs)
                self.sublayers = sublayers

            def call(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                prev_input = x
                for layer in self.sublayers:
                    prev_input = layer(prev_input)
                return prev_input

            def get_config(self):
                if False:
                    i = 10
                    return i + 15
                config = super().get_config()
                config['sublayers'] = serialization_lib.serialize_keras_object(self.sublayers)
                return config

            @classmethod
            def from_config(cls, config):
                if False:
                    print('Hello World!')
                config['sublayers'] = serialization_lib.deserialize_keras_object(config['sublayers'])
                return cls(**config)

        @object_registration.register_keras_serializable(package='Foo')
        class RegisteredSubLayer(layers.Layer):

            def call(self, x):
                if False:
                    return 10
                return x
        loaded_model = legacy_h5_format.load_model_from_hdf5(temp_filepath, custom_objects={'MyLayer': MyLayer})
        loaded_layer = loaded_model.layers[0]
        output = loaded_model(x)
        self.assertIsInstance(loaded_layer.sublayers[0], layers.Dense)
        self.assertEqual(loaded_layer.sublayers[0].name, 'MyDense')
        self.assertIsInstance(loaded_layer.sublayers[1], RegisteredSubLayer)
        self.assertEqual(loaded_layer.sublayers[1].name, 'MySubLayer')
        self.assertAllClose(ref_output, output, atol=1e-05)

@pytest.mark.requires_trainable_backend
class DirectoryCreationTest(testing.TestCase):

    def DISABLED_test_directory_creation_on_save(self):
        if False:
            print('Hello World!')
        'Test if directory is created on model save.'
        model = get_sequential_model(keras)
        nested_dirpath = os.path.join(self.get_temp_dir(), 'dir1', 'dir2', 'dir3')
        filepath = os.path.join(nested_dirpath, 'model.h5')
        self.assertFalse(os.path.exists(nested_dirpath))
        legacy_h5_format.save_model_to_hdf5(model, filepath)
        self.assertTrue(os.path.exists(nested_dirpath))
        loaded_model = legacy_h5_format.load_model_from_hdf5(filepath)
        self.assertEqual(model.to_json(), loaded_model.to_json())