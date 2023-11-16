"""Tests for `tf.data.TextLineDataset`."""
import gzip
import os
import pathlib
import zlib
from absl.testing import parameterized
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.util import compat
try:
    import psutil
    psutil_import_succeeded = True
except ImportError:
    psutil_import_succeeded = False

class TextLineDatasetTestBase(test_base.DatasetTestBase):
    """Base class for setting up and testing TextLineDataset."""

    def _lineText(self, f, l):
        if False:
            print('Hello World!')
        return compat.as_bytes('%d: %d' % (f, l))

    def _createFiles(self, num_files, num_lines, crlf=False, compression_type=None):
        if False:
            print('Hello World!')
        filenames = []
        for i in range(num_files):
            fn = os.path.join(self.get_temp_dir(), 'text_line.%d.txt' % i)
            filenames.append(fn)
            contents = []
            for j in range(num_lines):
                contents.append(self._lineText(i, j))
                if j + 1 != num_lines or i == 0:
                    contents.append(b'\r\n' if crlf else b'\n')
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

class TextLineDatasetTest(TextLineDatasetTestBase, parameterized.TestCase):

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(compression_type=[None, 'GZIP', 'ZLIB'])))
    def testBasic(self, compression_type):
        if False:
            for i in range(10):
                print('nop')
        test_filenames = self._createFiles(2, 5, crlf=True, compression_type=compression_type)

        def dataset_fn(filenames, num_epochs, batch_size=None):
            if False:
                i = 10
                return i + 15
            repeat_dataset = readers.TextLineDataset(filenames, compression_type=compression_type).repeat(num_epochs)
            if batch_size:
                return repeat_dataset.batch(batch_size)
            return repeat_dataset
        expected_output = [self._lineText(0, i) for i in range(5)]
        self.assertDatasetProduces(dataset_fn([test_filenames[0]], 1), expected_output=expected_output)
        self.assertDatasetProduces(dataset_fn([test_filenames[1]], 1), expected_output=[self._lineText(1, i) for i in range(5)])
        expected_output = [self._lineText(0, i) for i in range(5)]
        expected_output.extend((self._lineText(1, i) for i in range(5)))
        self.assertDatasetProduces(dataset_fn(test_filenames, 1), expected_output=expected_output)
        expected_output = [self._lineText(0, i) for i in range(5)]
        expected_output.extend((self._lineText(1, i) for i in range(5)))
        self.assertDatasetProduces(dataset_fn(test_filenames, 10), expected_output=expected_output * 10)
        self.assertDatasetProduces(dataset_fn(test_filenames, 10, 5), expected_output=[[self._lineText(0, i) for i in range(5)], [self._lineText(1, i) for i in range(5)]] * 10)

    @combinations.generate(test_base.default_test_combinations())
    def testParallelRead(self):
        if False:
            return 10
        test_filenames = self._createFiles(10, 10)
        files = dataset_ops.Dataset.from_tensor_slices(test_filenames).repeat(10)
        expected_output = []
        for j in range(10):
            expected_output.extend((self._lineText(j, i) for i in range(10)))
        dataset = readers.TextLineDataset(files, num_parallel_reads=4)
        self.assertDatasetProduces(dataset, expected_output=expected_output * 10, assert_items_equal=True)

    @combinations.generate(test_base.default_test_combinations())
    def testBuffering(self):
        if False:
            while True:
                i = 10
        test_filenames = self._createFiles(2, 5, crlf=True)
        repeat_dataset = readers.TextLineDataset(test_filenames, buffer_size=10)
        expected_output = []
        for j in range(2):
            expected_output.extend([self._lineText(j, i) for i in range(5)])
        self.assertDatasetProduces(repeat_dataset, expected_output=expected_output)

    @combinations.generate(test_base.eager_only_combinations())
    def testIteratorResourceCleanup(self):
        if False:
            for i in range(10):
                print('nop')
        filename = os.path.join(self.get_temp_dir(), 'text.txt')
        with open(filename, 'wt') as f:
            for i in range(3):
                f.write('%d\n' % (i,))
        first_iterator = iter(readers.TextLineDataset(filename))
        self.assertEqual(b'0', next(first_iterator).numpy())
        second_iterator = iter(readers.TextLineDataset(filename))
        self.assertEqual(b'0', next(second_iterator).numpy())
        different_kernel_iterator = iter(readers.TextLineDataset(filename).repeat().batch(16))
        self.assertEqual([16], next(different_kernel_iterator).shape)
        del first_iterator
        del second_iterator
        del different_kernel_iterator
        if not psutil_import_succeeded:
            self.skipTest("psutil is required to check that we've closed our files.")
        open_files = psutil.Process().open_files()
        self.assertNotIn(filename, [open_file.path for open_file in open_files])

    @combinations.generate(test_base.default_test_combinations())
    def testPathlib(self):
        if False:
            return 10
        files = self._createFiles(1, 5)
        files = [pathlib.Path(f) for f in files]
        expected_output = [self._lineText(0, i) for i in range(5)]
        ds = readers.TextLineDataset(files)
        self.assertDatasetProduces(ds, expected_output=expected_output, assert_items_equal=True)

    @combinations.generate(test_base.default_test_combinations())
    def testName(self):
        if False:
            while True:
                i = 10
        files = self._createFiles(1, 5)
        expected_output = [self._lineText(0, i) for i in range(5)]
        ds = readers.TextLineDataset(files, name='text_line_dataset')
        self.assertDatasetProduces(ds, expected_output=expected_output, assert_items_equal=True)

    @combinations.generate(test_base.default_test_combinations())
    def testEmptyFileList(self):
        if False:
            print('Hello World!')
        dataset = readers.TextLineDataset(filenames=[])
        self.assertDatasetProduces(dataset, [])

    @combinations.generate(test_base.default_test_combinations())
    def testFileDoesNotExist(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = readers.TextLineDataset(filenames=['File not exist'])
        with self.assertRaisesRegex(errors.NotFoundError, 'No such file or directory'):
            self.getDatasetOutput(dataset)

    @combinations.generate(test_base.default_test_combinations())
    def testFileNamesMustBeStrings(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(TypeError, 'The `filenames` argument must contain `tf.string` elements. Got `tf.int32` elements.'):
            readers.TextLineDataset(filenames=0)

    @combinations.generate(test_base.default_test_combinations())
    def testFileNamesDatasetMustContainStrings(self):
        if False:
            return 10
        with self.assertRaisesRegex(TypeError, 'The `filenames` argument must contain `tf.string` elements. Got a dataset of `tf.int32` elements.'):
            filenames = dataset_ops.Dataset.from_tensors(0)
            readers.TextLineDataset(filenames)

    @combinations.generate(test_base.default_test_combinations())
    def testFileNamesMustBeScalars(self):
        if False:
            return 10
        with self.assertRaisesRegex(TypeError, 'The `filenames` argument must contain `tf.string` elements of shape \\[\\] \\(i.e. scalars\\).'):
            filenames = dataset_ops.Dataset.from_tensors([['File 1', 'File 2'], ['File 3', 'File 4']])
            readers.TextLineDataset(filenames)

class TextLineDatasetCheckpointTest(TextLineDatasetTestBase, checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

    def _build_iterator_graph(self, test_filenames, compression_type=None):
        if False:
            print('Hello World!')
        return readers.TextLineDataset(test_filenames, compression_type=compression_type, buffer_size=10)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), checkpoint_test_base.default_test_combinations(), combinations.combine(compression_type=[None, 'GZIP', 'ZLIB'])))
    def test(self, verify_fn, compression_type):
        if False:
            return 10
        num_files = 5
        lines_per_file = 5
        num_outputs = num_files * lines_per_file
        test_filenames = self._createFiles(num_files, lines_per_file, crlf=True, compression_type=compression_type)
        verify_fn(self, lambda : self._build_iterator_graph(test_filenames, compression_type), num_outputs)
if __name__ == '__main__':
    test.main()