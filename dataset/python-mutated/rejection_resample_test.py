"""Tests for `tf.data.Dataset.rejection_resample()`."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat

class RejectionResampleTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(initial_known=[True, False])))
    def testDistribution(self, initial_known):
        if False:
            for i in range(10):
                print('nop')
        classes = np.random.randint(5, size=(10000,))
        target_dist = [0.9, 0.05, 0.05, 0.0, 0.0]
        initial_dist = [0.2] * 5 if initial_known else None
        classes = math_ops.cast(classes, dtypes.int64)
        dataset = dataset_ops.Dataset.from_tensor_slices(classes).shuffle(200, seed=21).map(lambda c: (c, string_ops.as_string(c))).repeat()
        get_next = self.getNext(dataset.rejection_resample(target_dist=target_dist, initial_dist=initial_dist, class_func=lambda c, _: c, seed=27), requires_initialization=True)
        returned = []
        while len(returned) < 2000:
            returned.append(self.evaluate(get_next()))
        (returned_classes, returned_classes_and_data) = zip(*returned)
        (_, returned_data) = zip(*returned_classes_and_data)
        self.assertAllEqual([compat.as_bytes(str(c)) for c in returned_classes], returned_data)
        total_returned = len(returned_classes)
        class_counts = np.array([len([True for v in returned_classes if v == c]) for c in range(5)])
        returned_dist = class_counts / total_returned
        self.assertAllClose(target_dist, returned_dist, atol=0.01)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(only_initial_dist=[True, False])))
    def testEdgeCasesSampleFromInitialDataset(self, only_initial_dist):
        if False:
            i = 10
            return i + 15
        init_dist = [0.5, 0.5]
        target_dist = [0.5, 0.5] if only_initial_dist else [0.0, 1.0]
        num_classes = len(init_dist)
        num_samples = 100
        data_np = np.random.choice(num_classes, num_samples, p=init_dist)
        dataset = dataset_ops.Dataset.from_tensor_slices(data_np)
        dataset = dataset.rejection_resample(class_func=lambda x: x, target_dist=target_dist, initial_dist=init_dist)
        get_next = self.getNext(dataset)
        returned = []
        with self.assertRaises(errors.OutOfRangeError):
            while True:
                returned.append(self.evaluate(get_next()))

    @combinations.generate(test_base.default_test_combinations())
    def testRandomClasses(self):
        if False:
            i = 10
            return i + 15
        init_dist = [0.25, 0.25, 0.25, 0.25]
        target_dist = [0.0, 0.0, 0.0, 1.0]
        num_classes = len(init_dist)
        num_samples = 100
        data_np = np.random.choice(num_classes, num_samples, p=init_dist)
        dataset = dataset_ops.Dataset.from_tensor_slices(data_np)

        def _remap_fn(_):
            if False:
                print('Hello World!')
            return math_ops.cast(random_ops.random_uniform([1]) * num_classes, dtypes.int32)[0]
        dataset = dataset.map(_remap_fn)
        dataset = dataset.rejection_resample(class_func=lambda x: x, target_dist=target_dist, initial_dist=init_dist)
        get_next = self.getNext(dataset)
        returned = []
        with self.assertRaises(errors.OutOfRangeError):
            while True:
                returned.append(self.evaluate(get_next()))
        (classes, _) = zip(*returned)
        bincount = np.bincount(np.array(classes), minlength=num_classes).astype(np.float32) / len(classes)
        self.assertAllClose(target_dist, bincount, atol=0.01)

    @combinations.generate(test_base.default_test_combinations())
    def testExhaustion(self):
        if False:
            return 10
        init_dist = [0.5, 0.5]
        target_dist = [0.9, 0.1]
        dataset = dataset_ops.Dataset.range(10000)
        dataset = dataset.rejection_resample(class_func=lambda x: x % 2, target_dist=target_dist, initial_dist=init_dist)
        get_next = self.getNext(dataset, requires_initialization=True)
        returned = []
        with self.assertRaises(errors.OutOfRangeError):
            while True:
                returned.append(self.evaluate(get_next()))
        (classes, _) = zip(*returned)
        bincount = np.bincount(np.array(classes), minlength=len(init_dist)).astype(np.float32) / len(classes)
        self.assertAllClose(target_dist, bincount, atol=0.01)

    @parameterized.parameters(('float32', 'float64'), ('float64', 'float32'), ('float64', 'float64'), ('float64', None))
    def testOtherDtypes(self, target_dtype, init_dtype):
        if False:
            print('Hello World!')
        target_dist = np.array([0.5, 0.5], dtype=target_dtype)
        if init_dtype is None:
            init_dist = None
        else:
            init_dist = np.array([0.5, 0.5], dtype=init_dtype)
        dataset = dataset_ops.Dataset.range(10)
        dataset = dataset.rejection_resample(class_func=lambda x: x % 2, target_dist=target_dist, initial_dist=init_dist)
        get_next = self.getNext(dataset, requires_initialization=True)
        self.evaluate(get_next())
if __name__ == '__main__':
    test.main()