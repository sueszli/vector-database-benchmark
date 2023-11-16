"""Tests for `tf.data.experimental.TFRecordWriter`."""
import os
from absl.testing import parameterized
from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.experimental.ops import writers
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import python_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat

class TFRecordWriterTest(test_base.DatasetTestBase, parameterized.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(TFRecordWriterTest, self).setUp()
        self._num_records = 8

    def writer_fn(self, filename, compression_type=''):
        if False:
            return 10
        input_dataset = readers.TFRecordDataset([filename], compression_type)
        return writers.TFRecordWriter(self._outputFilename(), compression_type).write(input_dataset)

    def _record(self, i):
        if False:
            return 10
        return compat.as_bytes('Record %d' % i)

    def _createFile(self, options=None):
        if False:
            while True:
                i = 10
        filename = self._inputFilename()
        writer = python_io.TFRecordWriter(filename, options)
        for i in range(self._num_records):
            writer.write(self._record(i))
        writer.close()
        return filename

    def _inputFilename(self):
        if False:
            return 10
        return os.path.join(self.get_temp_dir(), 'tf_record.in.txt')

    def _outputFilename(self):
        if False:
            return 10
        return os.path.join(self.get_temp_dir(), 'tf_record.out.txt')

    @combinations.generate(test_base.default_test_combinations())
    def testWrite(self):
        if False:
            i = 10
            return i + 15
        self.evaluate(self.writer_fn(self._createFile()))
        for (i, r) in enumerate(tf_record.tf_record_iterator(self._outputFilename())):
            self.assertAllEqual(self._record(i), r)

    @combinations.generate(test_base.default_test_combinations())
    def testWriteZLIB(self):
        if False:
            i = 10
            return i + 15
        options = tf_record.TFRecordOptions(tf_record.TFRecordCompressionType.ZLIB)
        self.evaluate(self.writer_fn(self._createFile(options), compression_type='ZLIB'))
        for (i, r) in enumerate(tf_record.tf_record_iterator(self._outputFilename(), options=options)):
            self.assertAllEqual(self._record(i), r)

    @combinations.generate(test_base.default_test_combinations())
    def testWriteGZIP(self):
        if False:
            i = 10
            return i + 15
        options = tf_record.TFRecordOptions(tf_record.TFRecordCompressionType.GZIP)
        self.evaluate(self.writer_fn(self._createFile(options), compression_type='GZIP'))
        for (i, r) in enumerate(tf_record.tf_record_iterator(self._outputFilename(), options=options)):
            self.assertAllEqual(self._record(i), r)

    @combinations.generate(test_base.default_test_combinations())
    def testFailDataset(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            writers.TFRecordWriter(self._outputFilename(), '').write('whoops')

    @combinations.generate(test_base.default_test_combinations())
    def testFailDType(self):
        if False:
            print('Hello World!')
        input_dataset = dataset_ops.Dataset.from_tensors(10)
        with self.assertRaises(TypeError):
            writers.TFRecordWriter(self._outputFilename(), '').write(input_dataset)

    @combinations.generate(test_base.default_test_combinations())
    def testFailShape(self):
        if False:
            i = 10
            return i + 15
        input_dataset = dataset_ops.Dataset.from_tensors([['hello'], ['world']])
        with self.assertRaises(TypeError):
            writers.TFRecordWriter(self._outputFilename(), '').write(input_dataset)

    @combinations.generate(test_base.default_test_combinations())
    def testSideEffect(self):
        if False:
            while True:
                i = 10

        def writer_fn():
            if False:
                return 10
            input_dataset = readers.TFRecordDataset(self._createFile())
            return writers.TFRecordWriter(self._outputFilename()).write(input_dataset)

        @def_function.function
        def fn():
            if False:
                for i in range(10):
                    print('nop')
            _ = writer_fn()
            return 'hello'
        self.assertEqual(self.evaluate(fn()), b'hello')
        for (i, r) in enumerate(tf_record.tf_record_iterator(self._outputFilename())):
            self.assertAllEqual(self._record(i), r)

    @combinations.generate(test_base.default_test_combinations())
    def testShard(self):
        if False:
            i = 10
            return i + 15
        filename = self._createFile()
        dataset = readers.TFRecordDataset([filename])

        def reduce_func(key, dataset):
            if False:
                while True:
                    i = 10
            shard_filename = string_ops.string_join([filename, string_ops.as_string(key)])
            writer = writers.TFRecordWriter(shard_filename)
            writer.write(dataset.map(lambda _, x: x))
            return dataset_ops.Dataset.from_tensors(shard_filename)
        dataset = dataset.enumerate()
        dataset = dataset.apply(grouping.group_by_window(lambda i, _: i % 2, reduce_func, dtypes.int64.max))
        get_next = self.getNext(dataset)
        for i in range(2):
            shard_filename = (filename + str(i)).encode()
            self.assertEqual(self.evaluate(get_next()), shard_filename)
            for (j, r) in enumerate(tf_record.tf_record_iterator(shard_filename)):
                self.assertAllEqual(self._record(i + 2 * j), r)
if __name__ == '__main__':
    test.main()