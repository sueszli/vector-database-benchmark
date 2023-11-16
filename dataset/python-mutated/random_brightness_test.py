import numpy as np
import pytest
from tensorflow import data as tf_data
from keras import backend
from keras import layers
from keras import testing

class RandomBrightnessTest(testing.TestCase):

    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        if False:
            i = 10
            return i + 15
        self.run_layer_test(layers.RandomBrightness, init_kwargs={'factor': 0.75, 'value_range': (20, 200), 'seed': 1}, input_shape=(8, 3, 4, 3), supports_masking=False, expected_output_shape=(8, 3, 4, 3))

    def test_random_brightness_inference(self):
        if False:
            i = 10
            return i + 15
        seed = 3481
        layer = layers.RandomBrightness([0, 1.0])
        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)

    def test_output(self):
        if False:
            while True:
                i = 10
        seed = 2390
        layer = layers.RandomBrightness([0, 1.0])
        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = backend.convert_to_numpy(layer(inputs))
        diff = output - inputs
        diff = backend.convert_to_numpy(diff)
        self.assertTrue(np.amin(diff) >= 0)
        self.assertTrue(np.mean(diff) > 0)
        layer = layers.RandomBrightness([-1.0, 0.0])
        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = backend.convert_to_numpy(layer(inputs))
        diff = output - inputs
        self.assertTrue(np.amax(diff) <= 0)
        self.assertTrue(np.mean(diff) < 0)

    def test_tf_data_compatibility(self):
        if False:
            print('Hello World!')
        layer = layers.RandomBrightness(factor=0.5, seed=1337)
        input_data = np.random.random((2, 8, 8, 3))
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output.numpy()