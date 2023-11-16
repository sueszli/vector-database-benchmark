"""Tests for `tf.data.FixedLengthRecordDataset`."""
import gzip
import os
import pathlib
import zlib
from absl.testing import parameterized
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.util import compat

class FixedLengthRecordDatasetTestBase(test_base.DatasetTestBase):
    """Base class for setting up and testing FixedLengthRecordDataset."""

    def setUp(self):
        if False:
            print('Hello World!')
        super(FixedLengthRecordDatasetTestBase, self).setUp()
        self._num_files = 2
        self._num_records = 7
        self._header_bytes = 5
        self._record_bytes = 3
        self._footer_bytes = 2

    def _record(self, f, r):
        if False:
            print('Hello World!')
        return compat.as_bytes(str(f * 2 + r) * self._record_bytes)

    def _createFiles(self, compression_type=None):
        if False:
            i = 10
            return i + 15
        filenames = []
        for i in range(self._num_files):
            fn = os.path.join(self.get_temp_dir(), 'fixed_length_record.%d.txt' % i)
            filenames.append(fn)
            contents = []
            contents.append(b'H' * self._header_bytes)
            for j in range(self._num_records):
                contents.append(self._record(i, j))
            contents.append(b'F' * self._footer_bytes)
            contents = b''.join(contents)
            if not compression_type:
                with open(fn, 'wb') as f:
                    f.write(contents)
            elif compression_type == 'GZIP':
                with gzip.GzipFile(fn, 'wb') as f:
                    f.write(contents)
            elif compression_type == 'ZLIB':
                contents = zlib.compress(contents)
                with open(fn, 'wb') as f:
                    f.write(contents)
            else:
                raise ValueError('Unsupported compression_type', compression_type)
        return filenames

class FixedLengthRecordDatasetTest(FixedLengthRecordDatasetTestBase, parameterized.TestCase):

    def _test(self, compression_type=None):
        if False:
            while True:
                i = 10
        test_filenames = self._createFiles(compression_type=compression_type)

        def dataset_fn(filenames, num_epochs, batch_size=None):
            if False:
                for i in range(10):
                    print('nop')
            repeat_dataset = readers.FixedLengthRecordDataset(filenames, self._record_bytes, self._header_bytes, self._footer_bytes, compression_type=compression_type).repeat(num_epochs)
            if batch_size:
                return repeat_dataset.batch(batch_size)
            return repeat_dataset
        self.assertDatasetProduces(dataset_fn([test_filenames[0]], 1), expected_output=[self._record(0, i) for i in range(self._num_records)])
        self.assertDatasetProduces(dataset_fn([test_filenames[1]], 1), expected_output=[self._record(1, i) for i in range(self._num_records)])
        expected_output = []
        for j in range(self._num_files):
            expected_output.extend([self._record(j, i) for i in range(self._num_records)])
        self.assertDatasetProduces(dataset_fn(test_filenames, 1), expected_output=expected_output)
        get_next = self.getNext(dataset_fn(test_filenames, 10))
        for _ in range(10):
            for j in range(self._num_files):
                for i in range(self._num_records):
                    self.assertEqual(self._record(j, i), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())
        get_next = self.getNext(dataset_fn(test_filenames, 10, self._num_records))
        for _ in range(10):
            for j in range(self._num_files):
                self.assertAllEqual([self._record(j, i) for i in range(self._num_records)], self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testNoCompression(self):
        if False:
            print('Hello World!')
        self._test()

    @combinations.generate(test_base.default_test_combinations())
    def testGzipCompression(self):
        if False:
            while True:
                i = 10
        self._test(compression_type='GZIP')

    @combinations.generate(test_base.default_test_combinations())
    def testZlibCompression(self):
        if False:
            print('Hello World!')
        self._test(compression_type='ZLIB')

    @combinations.generate(test_base.default_test_combinations())
    def testBuffering(self):
        if False:
            return 10
        test_filenames = self._createFiles()
        dataset = readers.FixedLengthRecordDataset(test_filenames, self._record_bytes, self._header_bytes, self._footer_bytes, buffer_size=10)
        expected_output = []
        for j in range(self._num_files):
            expected_output.extend([self._record(j, i) for i in range(self._num_records)])
        self.assertDatasetProduces(dataset, expected_output=expected_output)

    @combinations.generate(test_base.default_test_combinations())
    def testParallelRead(self):
        if False:
            return 10
        test_filenames = self._createFiles()
        dataset = readers.FixedLengthRecordDataset(test_filenames, self._record_bytes, self._header_bytes, self._footer_bytes, buffer_size=10, num_parallel_reads=4)
        expected_output = []
        for j in range(self._num_files):
            expected_output.extend([self._record(j, i) for i in range(self._num_records)])
        self.assertDatasetProduces(dataset, expected_output=expected_output, assert_items_equal=True)

    @combinations.generate(test_base.default_test_combinations())
    def testWrongSize(self):
        if False:
            i = 10
            return i + 15
        test_filenames = self._createFiles()
        dataset = readers.FixedLengthRecordDataset(test_filenames, self._record_bytes + 1, self._header_bytes, self._footer_bytes, buffer_size=10)
        self.assertDatasetProduces(dataset, expected_error=(errors.InvalidArgumentError, 'Excluding the header \\(5 bytes\\) and footer \\(2 bytes\\), input file \\".*fixed_length_record.0.txt\\" has body length 21 bytes, which is not an exact multiple of the record length \\(4 bytes\\).'))

    @combinations.generate(test_base.default_test_combinations())
    def testPathlib(self):
        if False:
            print('Hello World!')
        test_filenames = self._createFiles()
        test_filenames = [pathlib.Path(f) for f in test_filenames]
        dataset = readers.FixedLengthRecordDataset(test_filenames, self._record_bytes, self._header_bytes, self._footer_bytes, buffer_size=10, num_parallel_reads=4)
        expected_output = []
        for j in range(self._num_files):
            expected_output.extend([self._record(j, i) for i in range(self._num_records)])
        self.assertDatasetProduces(dataset, expected_output=expected_output, assert_items_equal=True)

    @combinations.generate(test_base.default_test_combinations())
    def testName(self):
        if False:
            return 10
        test_filenames = self._createFiles()
        dataset = readers.FixedLengthRecordDataset(test_filenames, self._record_bytes, self._header_bytes, self._footer_bytes, name='fixed_length_record_dataset')
        expected_output = []
        for j in range(self._num_files):
            expected_output.extend([self._record(j, i) for i in range(self._num_records)])
        self.assertDatasetProduces(dataset, expected_output=expected_output)

class FixedLengthRecordDatasetCheckpointTest(FixedLengthRecordDatasetTestBase, checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

    def _build_dataset(self, num_epochs, compression_type=None):
        if False:
            return 10
        filenames = self._createFiles()
        return readers.FixedLengthRecordDataset(filenames, self._record_bytes, self._header_bytes, self._footer_bytes).repeat(num_epochs)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), checkpoint_test_base.default_test_combinations()))
    def test(self, verify_fn):
        if False:
            i = 10
            return i + 15
        num_epochs = 5
        num_outputs = num_epochs * self._num_files * self._num_records
        verify_fn(self, lambda : self._build_dataset(num_epochs), num_outputs)
if __name__ == '__main__':
    test.main()