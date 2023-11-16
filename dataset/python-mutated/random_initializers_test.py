import numpy as np
from keras import backend
from keras import initializers
from keras import testing
from keras import utils

class InitializersTest(testing.TestCase):

    def test_random_normal(self):
        if False:
            for i in range(10):
                print('nop')
        utils.set_random_seed(1337)
        shape = (25, 20)
        mean = 0.0
        stddev = 1.0
        seed = 1234
        initializer = initializers.RandomNormal(mean=mean, stddev=stddev, seed=seed)
        values = initializer(shape=shape)
        self.assertEqual(initializer.mean, mean)
        self.assertEqual(initializer.stddev, stddev)
        self.assertEqual(initializer.seed, seed)
        self.assertEqual(values.shape, shape)
        self.assertAllClose(np.std(backend.convert_to_numpy(values)), stddev, atol=0.1)
        self.run_class_serialization_test(initializer)
        initializer = initializers.RandomNormal(mean=mean, stddev=stddev, seed=1337)
        values = initializer(shape=shape)
        next_values = initializer(shape=shape)
        self.assertAllClose(values, next_values)
        initializer = initializers.RandomNormal(mean=mean, stddev=stddev, seed=backend.random.SeedGenerator(1337))
        values = initializer(shape=shape)
        next_values = initializer(shape=shape)
        self.assertNotAllClose(values, next_values)
        initializer = initializers.RandomNormal(mean=mean, stddev=stddev, seed=backend.random.SeedGenerator(1337))
        values = initializer(shape=shape)
        initializer = initializers.RandomNormal(mean=mean, stddev=stddev, seed=None)
        values = initializer(shape=shape)
        cloned_initializer = initializers.RandomNormal.from_config(initializer.get_config())
        new_values = cloned_initializer(shape=shape)
        self.assertNotAllClose(values, new_values)
        initializer = initializers.RandomNormal(mean=mean, stddev=stddev, seed=1337)
        values = initializer(shape=shape)
        cloned_initializer = initializers.RandomNormal.from_config(initializer.get_config())
        new_values = cloned_initializer(shape=shape)
        self.assertAllClose(values, new_values)

    def test_random_uniform(self):
        if False:
            while True:
                i = 10
        shape = (5, 5)
        minval = -1.0
        maxval = 1.0
        seed = 1234
        initializer = initializers.RandomUniform(minval=minval, maxval=maxval, seed=seed)
        values = initializer(shape=shape)
        self.assertEqual(initializer.minval, minval)
        self.assertEqual(initializer.maxval, maxval)
        self.assertEqual(initializer.seed, seed)
        self.assertEqual(values.shape, shape)
        values = backend.convert_to_numpy(values)
        self.assertGreaterEqual(np.min(values), minval)
        self.assertLess(np.max(values), maxval)
        self.run_class_serialization_test(initializer)

    def test_variance_scaling(self):
        if False:
            return 10
        utils.set_random_seed(1337)
        shape = (25, 20)
        scale = 2.0
        seed = 1234
        initializer = initializers.VarianceScaling(scale=scale, seed=seed, mode='fan_in')
        values = initializer(shape=shape)
        self.assertEqual(initializer.scale, scale)
        self.assertEqual(initializer.seed, seed)
        self.assertEqual(values.shape, shape)
        self.assertAllClose(np.std(backend.convert_to_numpy(values)), np.sqrt(scale / 25), atol=0.1)
        self.run_class_serialization_test(initializer)
        initializer = initializers.VarianceScaling(scale=scale, seed=seed, mode='fan_out')
        values = initializer(shape=shape)
        self.assertEqual(initializer.scale, scale)
        self.assertEqual(initializer.seed, seed)
        self.assertEqual(values.shape, shape)
        self.assertAllClose(np.std(backend.convert_to_numpy(values)), np.sqrt(scale / 20), atol=0.1)
        self.run_class_serialization_test(initializer)

    def test_orthogonal_initializer(self):
        if False:
            i = 10
            return i + 15
        shape = (5, 5)
        gain = 2.0
        seed = 1234
        initializer = initializers.OrthogonalInitializer(gain=gain, seed=seed)
        values = initializer(shape=shape)
        self.assertEqual(initializer.seed, seed)
        self.assertEqual(initializer.gain, gain)
        self.assertEqual(values.shape, shape)
        array = backend.convert_to_numpy(values)
        for column in array.T:
            self.assertAlmostEqual(np.linalg.norm(column), gain * 1.0)
        for i in range(array.shape[-1]):
            for j in range(i + 1, array.shape[-1]):
                self.assertAlmostEqual(np.dot(array[..., i], array[..., j]), 0.0)
        self.run_class_serialization_test(initializer)

    def test_get_method(self):
        if False:
            return 10
        obj = initializers.get('glorot_normal')
        self.assertTrue(obj, initializers.GlorotNormal)
        obj = initializers.get(None)
        self.assertEqual(obj, None)
        with self.assertRaises(ValueError):
            initializers.get('typo')

    def test_variance_scaling_invalid_scale(self):
        if False:
            print('Hello World!')
        seed = 1234
        with self.assertRaisesRegex(ValueError, 'Argument `scale` must be positive float.'):
            initializers.VarianceScaling(scale=-1.0, seed=seed, mode='fan_in')

    def test_variance_scaling_invalid_mode(self):
        if False:
            i = 10
            return i + 15
        scale = 2.0
        seed = 1234
        with self.assertRaisesRegex(ValueError, 'Invalid `mode` argument:'):
            initializers.VarianceScaling(scale=scale, seed=seed, mode='invalid_mode')

    def test_variance_scaling_invalid_distribution(self):
        if False:
            return 10
        scale = 2.0
        seed = 1234
        with self.assertRaisesRegex(ValueError, 'Invalid `distribution` argument:'):
            initializers.VarianceScaling(scale=scale, seed=seed, mode='fan_in', distribution='invalid_dist')