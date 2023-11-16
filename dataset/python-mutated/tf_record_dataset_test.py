"""Tests for `tf.data.TFRecordDataset`."""
import gzip
import os
import pathlib
import zlib
from absl.testing import parameterized
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.kernel_tests import tf_record_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test

class TFRecordDatasetTest(tf_record_test_base.TFRecordTestBase, parameterized.TestCase):

    def _dataset_factory(self, filenames, compression_type='', num_epochs=1, batch_size=None):
        if False:
            i = 10
            return i + 15
        repeat_dataset = readers.TFRecordDataset(filenames, compression_type).repeat(num_epochs)
        if batch_size:
            return repeat_dataset.batch(batch_size)
        return repeat_dataset

    @combinations.generate(test_base.default_test_combinations())
    def testConstructorErrorsTensorInput(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(TypeError, 'The `filenames` argument must contain `tf.string` elements. Got `tf.int32` elements.'):
            readers.TFRecordDataset([1, 2, 3])
        with self.assertRaisesRegex(TypeError, 'The `filenames` argument must contain `tf.string` elements. Got `tf.int32` elements.'):
            readers.TFRecordDataset(constant_op.constant([1, 2, 3]))
        with self.assertRaises(Exception):
            readers.TFRecordDataset(object())

    @combinations.generate(test_base.default_test_combinations())
    def testReadOneEpoch(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = self._dataset_factory(self._filenames[0])
        self.assertDatasetProduces(dataset, expected_output=[self._record(0, i) for i in range(self._num_records)])
        dataset = self._dataset_factory(self._filenames[1])
        self.assertDatasetProduces(dataset, expected_output=[self._record(1, i) for i in range(self._num_records)])
        dataset = self._dataset_factory(self._filenames)
        expected_output = []
        for j in range(self._num_files):
            expected_output.extend([self._record(j, i) for i in range(self._num_records)])
        self.assertDatasetProduces(dataset, expected_output=expected_output)

    @combinations.generate(test_base.default_test_combinations())
    def testReadTenEpochs(self):
        if False:
            i = 10
            return i + 15
        dataset = self._dataset_factory(self._filenames, num_epochs=10)
        expected_output = []
        for j in range(self._num_files):
            expected_output.extend([self._record(j, i) for i in range(self._num_records)])
        self.assertDatasetProduces(dataset, expected_output=expected_output * 10)

    @combinations.generate(test_base.default_test_combinations())
    def testReadTenEpochsOfBatches(self):
        if False:
            while True:
                i = 10
        dataset = self._dataset_factory(self._filenames, num_epochs=10, batch_size=self._num_records)
        expected_output = []
        for j in range(self._num_files):
            expected_output.append([self._record(j, i) for i in range(self._num_records)])
        self.assertDatasetProduces(dataset, expected_output=expected_output * 10)

    @combinations.generate(test_base.default_test_combinations())
    def testReadZlibFiles(self):
        if False:
            while True:
                i = 10
        zlib_files = []
        for (i, fn) in enumerate(self._filenames):
            with open(fn, 'rb') as f:
                cdata = zlib.compress(f.read())
                zfn = os.path.join(self.get_temp_dir(), 'tfrecord_%s.z' % i)
                with open(zfn, 'wb') as f:
                    f.write(cdata)
                zlib_files.append(zfn)
        expected_output = []
        for j in range(self._num_files):
            expected_output.extend([self._record(j, i) for i in range(self._num_records)])
        dataset = self._dataset_factory(zlib_files, compression_type='ZLIB')
        self.assertDatasetProduces(dataset, expected_output=expected_output)

    @combinations.generate(test_base.default_test_combinations())
    def testReadGzipFiles(self):
        if False:
            while True:
                i = 10
        gzip_files = []
        for (i, fn) in enumerate(self._filenames):
            with open(fn, 'rb') as f:
                gzfn = os.path.join(self.get_temp_dir(), 'tfrecord_%s.gz' % i)
                with gzip.GzipFile(gzfn, 'wb') as gzf:
                    gzf.write(f.read())
                gzip_files.append(gzfn)
        expected_output = []
        for j in range(self._num_files):
            expected_output.extend([self._record(j, i) for i in range(self._num_records)])
        dataset = self._dataset_factory(gzip_files, compression_type='GZIP')
        self.assertDatasetProduces(dataset, expected_output=expected_output)

    @combinations.generate(test_base.default_test_combinations())
    def testReadWithBuffer(self):
        if False:
            print('Hello World!')
        one_mebibyte = 2 ** 20
        dataset = readers.TFRecordDataset(self._filenames, buffer_size=one_mebibyte)
        expected_output = []
        for j in range(self._num_files):
            expected_output.extend([self._record(j, i) for i in range(self._num_records)])
        self.assertDatasetProduces(dataset, expected_output=expected_output)

    @combinations.generate(test_base.default_test_combinations())
    def testReadFromDatasetOfFiles(self):
        if False:
            print('Hello World!')
        files = dataset_ops.Dataset.from_tensor_slices(self._filenames)
        expected_output = []
        for j in range(self._num_files):
            expected_output.extend([self._record(j, i) for i in range(self._num_records)])
        dataset = readers.TFRecordDataset(files)
        self.assertDatasetProduces(dataset, expected_output=expected_output)

    @combinations.generate(test_base.default_test_combinations())
    def testReadTenEpochsFromDatasetOfFilesInParallel(self):
        if False:
            return 10
        files = dataset_ops.Dataset.from_tensor_slices(self._filenames).repeat(10)
        expected_output = []
        for j in range(self._num_files):
            expected_output.extend([self._record(j, i) for i in range(self._num_records)])
        dataset = readers.TFRecordDataset(files, num_parallel_reads=4)
        self.assertDatasetProduces(dataset, expected_output=expected_output * 10, assert_items_equal=True)

    @combinations.generate(test_base.default_test_combinations())
    def testPathlib(self):
        if False:
            while True:
                i = 10
        files = [pathlib.Path(self._filenames[0])]
        expected_output = [self._record(0, i) for i in range(self._num_records)]
        ds = readers.TFRecordDataset(files)
        self.assertDatasetProduces(ds, expected_output=expected_output, assert_items_equal=True)

    @combinations.generate(test_base.default_test_combinations())
    def testName(self):
        if False:
            return 10
        files = [self._filenames[0]]
        expected_output = [self._record(0, i) for i in range(self._num_records)]
        ds = readers.TFRecordDataset(files, name='tf_record_dataset')
        self.assertDatasetProduces(ds, expected_output=expected_output, assert_items_equal=True)

class TFRecordDatasetCheckpointTest(tf_record_test_base.TFRecordTestBase, checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

    def make_dataset(self, num_epochs, compression_type=None, buffer_size=None, symbolic_checkpoint=False):
        if False:
            while True:
                i = 10
        filenames = self._createFiles()
        if compression_type == 'ZLIB':
            zlib_files = []
            for (i, fn) in enumerate(filenames):
                with open(fn, 'rb') as f:
                    cdata = zlib.compress(f.read())
                    zfn = os.path.join(self.get_temp_dir(), 'tfrecord_%s.z' % i)
                    with open(zfn, 'wb') as f:
                        f.write(cdata)
                    zlib_files.append(zfn)
            filenames = zlib_files
        elif compression_type == 'GZIP':
            gzip_files = []
            for (i, fn) in enumerate(self._filenames):
                with open(fn, 'rb') as f:
                    gzfn = os.path.join(self.get_temp_dir(), 'tfrecord_%s.gz' % i)
                    with gzip.GzipFile(gzfn, 'wb') as gzf:
                        gzf.write(f.read())
                    gzip_files.append(gzfn)
            filenames = gzip_files
        dataset = readers.TFRecordDataset(filenames, compression_type, buffer_size=buffer_size).repeat(num_epochs)
        if symbolic_checkpoint:
            options = options_lib.Options()
            options.experimental_symbolic_checkpoint = symbolic_checkpoint
            dataset = dataset.with_options(options)
        return dataset

    @combinations.generate(combinations.times(test_base.default_test_combinations(), checkpoint_test_base.default_test_combinations(), combinations.combine(symbolic_checkpoint=[True, False])))
    def test(self, verify_fn, symbolic_checkpoint):
        if False:
            for i in range(10):
                print('nop')
        num_epochs = 5
        num_outputs = num_epochs * self._num_files * self._num_records
        verify_fn(self, lambda : self.make_dataset(num_epochs, symbolic_checkpoint=symbolic_checkpoint), num_outputs)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), checkpoint_test_base.default_test_combinations(), combinations.combine(buffer_size=[0, 5])))
    def testBufferSize(self, verify_fn, buffer_size):
        if False:
            while True:
                i = 10
        num_epochs = 5
        num_outputs = num_epochs * self._num_files * self._num_records
        verify_fn(self, lambda : self.make_dataset(num_epochs, buffer_size=buffer_size), num_outputs)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), checkpoint_test_base.default_test_combinations(), combinations.combine(compression_type=[None, 'GZIP', 'ZLIB'])))
    def testCompressionTypes(self, verify_fn, compression_type):
        if False:
            for i in range(10):
                print('nop')
        num_epochs = 5
        num_outputs = num_epochs * self._num_files * self._num_records
        verify_fn(self, lambda : self.make_dataset(num_epochs, compression_type=compression_type), num_outputs)
if __name__ == '__main__':
    test.main()