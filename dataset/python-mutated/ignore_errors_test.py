"""Tests for `tf.data.Dataset.ignore_errors`."""
import os
import sys
from absl.testing import parameterized
import numpy as np
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat
_NUMPY_RANDOM_SEED = 42

class IgnoreErrorsTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(test_base.default_test_combinations())
    def testMapIgnoreError(self):
        if False:
            for i in range(10):
                print('nop')
        components = np.array([1.0, 2.0, 3.0, np.nan, 5.0]).astype(np.float32)
        dataset = dataset_ops.Dataset.from_tensor_slices(components).map(lambda x: array_ops.check_numerics(x, 'message')).ignore_errors()
        get_next = self.getNext(dataset)
        for x in [1.0, 2.0, 3.0, 5.0]:
            self.assertEqual(x, self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testIgnoreError_withLogWarning(self):
        if False:
            return 10
        components = np.array([1.0, 2.0, 3.0, np.nan, 5.0]).astype(np.float32)
        dataset = dataset_ops.Dataset.from_tensor_slices(components).map(lambda x: array_ops.check_numerics(x, 'message')).ignore_errors(log_warning=True)
        get_next = self.getNext(dataset)
        with self.captureWritesToStream(sys.stderr) as logged:
            for x in [1.0, 2.0, 3.0]:
                self.assertEqual(x, self.evaluate(get_next()))
            self.assertEqual(5.0, self.evaluate(get_next()))
        expected = 'Tensor had NaN values'
        self.assertIn(expected, logged.contents())
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testParallelMapIgnoreError(self):
        if False:
            while True:
                i = 10
        components = np.array([1.0, 2.0, 3.0, np.nan, 5.0]).astype(np.float32)
        dataset = dataset_ops.Dataset.from_tensor_slices(components).map(lambda x: array_ops.check_numerics(x, 'message'), num_parallel_calls=2).prefetch(2).ignore_errors()
        get_next = self.getNext(dataset)
        for x in [1.0, 2.0, 3.0, 5.0]:
            self.assertEqual(x, self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadFileIgnoreError(self):
        if False:
            return 10

        def write_string_to_file(value, filename):
            if False:
                i = 10
                return i + 15
            with open(filename, 'w') as f:
                f.write(value)
        filenames = [os.path.join(self.get_temp_dir(), 'file_%d.txt' % i) for i in range(5)]
        for filename in filenames:
            write_string_to_file(filename, filename)
        dataset = dataset_ops.Dataset.from_tensor_slices(filenames).map(io_ops.read_file, num_parallel_calls=2).prefetch(2).ignore_errors()
        get_next = self.getNext(dataset)
        for filename in filenames:
            self.assertEqual(compat.as_bytes(filename), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())
        os.remove(filenames[0])
        get_next = self.getNext(dataset)
        for filename in filenames[1:]:
            self.assertEqual(compat.as_bytes(filename), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testTFRecordDatasetIgnoreError(self):
        if False:
            print('Hello World!')
        filenames = []
        for i in range(5):
            fn = os.path.join(self.get_temp_dir(), 'tf_record.%d.txt' % i)
            filenames.append(fn)
            writer = python_io.TFRecordWriter(fn)
            for _ in range(10):
                writer.write(b'record')
            writer.close()
            with open(fn, 'a') as f:
                f.write('corrupted data')
        dataset = readers.TFRecordDataset(filenames).ignore_errors()
        get_next = self.getNext(dataset)
        for _ in filenames:
            for _ in range(10):
                self.assertEqual(b'record', self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testZipIgnoreError(self):
        if False:
            while True:
                i = 10
        a = dataset_ops.Dataset.from_tensor_slices([1.0, 2.0, 0.0, 4.0])
        b = a.map(lambda x: array_ops.check_numerics(1.0 / x, 'error'))
        dataset = dataset_ops.Dataset.zip((b, a)).ignore_errors()
        get_next = self.getNext(dataset)
        for x in [1.0, 2.0, 4.0]:
            self.assertEqual((1.0 / x, x), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testCardinality(self):
        if False:
            return 10
        ds = dataset_ops.Dataset.range(10).ignore_errors()
        self.assertEqual(self.evaluate(ds.cardinality()), dataset_ops.UNKNOWN)

class IgnoreErrorsCheckpointTest(checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

    def _build_ds(self):
        if False:
            return 10
        components = np.array([1.0, 2.0, 3.0, np.nan, 5.0]).astype(np.float32)
        dataset = dataset_ops.Dataset.from_tensor_slices(components)
        dataset = dataset.map(lambda x: array_ops.check_numerics(x, 'message'))
        dataset = dataset.ignore_errors()
        options = options_lib.Options()
        options.experimental_external_state_policy = options_lib.ExternalStatePolicy.IGNORE
        return dataset.with_options(options)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), checkpoint_test_base.default_test_combinations()))
    def test(self, verify_fn):
        if False:
            i = 10
            return i + 15
        verify_fn(self, self._build_ds, num_outputs=4)
if __name__ == '__main__':
    test.main()