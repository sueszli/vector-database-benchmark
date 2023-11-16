import numpy as np
from tensorflow import data as tf_data
from keras import backend
from keras import layers
from keras import testing

class StringLookupTest(testing.TestCase):

    def test_config(self):
        if False:
            return 10
        layer = layers.StringLookup(output_mode='int', vocabulary=['a', 'b', 'c'], oov_token='[OOV]', mask_token='[MASK]')
        self.run_class_serialization_test(layer)

    def test_adapt_flow(self):
        if False:
            while True:
                i = 10
        layer = layers.StringLookup(output_mode='int')
        layer.adapt(['a', 'a', 'a', 'b', 'b', 'c'])
        input_data = ['b', 'c', 'd']
        output = layer(input_data)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, np.array([2, 3, 0]))

    def test_fixed_vocabulary(self):
        if False:
            for i in range(10):
                print('nop')
        layer = layers.StringLookup(output_mode='int', vocabulary=['a', 'b', 'c'])
        input_data = ['b', 'c', 'd']
        output = layer(input_data)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, np.array([2, 3, 0]))

    def test_set_vocabulary(self):
        if False:
            i = 10
            return i + 15
        layer = layers.StringLookup(output_mode='int')
        layer.set_vocabulary(['a', 'b', 'c'])
        input_data = ['b', 'c', 'd']
        output = layer(input_data)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, np.array([2, 3, 0]))

    def test_tf_data_compatibility(self):
        if False:
            for i in range(10):
                print('nop')
        layer = layers.StringLookup(output_mode='int', vocabulary=['a', 'b', 'c'])
        input_data = ['b', 'c', 'd']
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(3).map(layer)
        for output in ds.take(1):
            output = output.numpy()
        self.assertAllClose(output, np.array([2, 3, 0]))