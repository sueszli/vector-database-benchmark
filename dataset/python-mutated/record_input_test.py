"""Tests for record_input_op."""
import os
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

class RecordInputOpTest(test.TestCase):

    def generateTestData(self, prefix, n, m, compression_type=tf_record.TFRecordCompressionType.NONE):
        if False:
            return 10
        options = tf_record.TFRecordOptions(compression_type)
        for i in range(n):
            f = os.path.join(self.get_temp_dir(), prefix + '.' + str(i))
            w = tf_record.TFRecordWriter(f, options=options)
            for j in range(m):
                w.write('{0:0{width}}'.format(i * m + j, width=10).encode('utf-8'))
        w.close()

    def testRecordInputSimple(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            self.generateTestData('basic', 1, 1)
            yield_op = data_flow_ops.RecordInput(file_pattern=os.path.join(self.get_temp_dir(), 'basic.*'), parallelism=1, buffer_size=1, batch_size=1, name='record_input').get_yield_op()
            self.assertEqual(self.evaluate(yield_op), b'0000000000')

    def testRecordInputSimpleGzip(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session() as sess:
            self.generateTestData('basic', 1, 1, compression_type=tf_record.TFRecordCompressionType.GZIP)
            yield_op = data_flow_ops.RecordInput(file_pattern=os.path.join(self.get_temp_dir(), 'basic.*'), parallelism=1, buffer_size=1, batch_size=1, name='record_input', compression_type=tf_record.TFRecordCompressionType.GZIP).get_yield_op()
            self.assertEqual(self.evaluate(yield_op), b'0000000000')

    def testRecordInputSimpleZlib(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session() as sess:
            self.generateTestData('basic', 1, 1, compression_type=tf_record.TFRecordCompressionType.ZLIB)
            yield_op = data_flow_ops.RecordInput(file_pattern=os.path.join(self.get_temp_dir(), 'basic.*'), parallelism=1, buffer_size=1, batch_size=1, name='record_input', compression_type=tf_record.TFRecordCompressionType.ZLIB).get_yield_op()
            self.assertEqual(self.evaluate(yield_op), b'0000000000')

    @test_util.run_deprecated_v1
    def testRecordInputEpochs(self):
        if False:
            while True:
                i = 10
        files = 100
        records_per_file = 100
        batches = 2
        with self.cached_session() as sess:
            self.generateTestData('basic', files, records_per_file)
            records = data_flow_ops.RecordInput(file_pattern=os.path.join(self.get_temp_dir(), 'basic.*'), parallelism=2, buffer_size=2000, batch_size=1, shift_ratio=0.33, seed=10, name='record_input', batches=batches)
            yield_op = records.get_yield_op()
            for _ in range(3):
                epoch_set = set()
                for _ in range(int(files * records_per_file / batches)):
                    op_list = self.evaluate(yield_op)
                    self.assertTrue(len(op_list) is batches)
                    for r in op_list:
                        self.assertTrue(r[0] not in epoch_set)
                        epoch_set.add(r[0])

    @test_util.run_deprecated_v1
    def testDoesNotDeadlock(self):
        if False:
            for i in range(10):
                print('nop')
        for _ in range(30):
            with self.cached_session() as sess:
                self.generateTestData('basic', 1, 1)
                records = data_flow_ops.RecordInput(file_pattern=os.path.join(self.get_temp_dir(), 'basic.*'), parallelism=1, buffer_size=100, batch_size=1, name='record_input')
                yield_op = records.get_yield_op()
                for _ in range(50):
                    self.evaluate(yield_op)

    @test_util.run_deprecated_v1
    def testEmptyGlob(self):
        if False:
            return 10
        with self.cached_session() as sess:
            record_input = data_flow_ops.RecordInput(file_pattern='foo')
            yield_op = record_input.get_yield_op()
            self.evaluate(variables.global_variables_initializer())
            with self.assertRaises(errors_impl.NotFoundError):
                self.evaluate(yield_op)

    @test_util.run_deprecated_v1
    def testBufferTooSmall(self):
        if False:
            return 10
        files = 10
        records_per_file = 10
        batches = 2
        with self.cached_session() as sess:
            self.generateTestData('basic', files, records_per_file)
            records = data_flow_ops.RecordInput(file_pattern=os.path.join(self.get_temp_dir(), 'basic.*'), parallelism=2, buffer_size=2000, batch_size=1, shift_ratio=0.33, seed=10, name='record_input', batches=batches)
            yield_op = records.get_yield_op()
            for _ in range(3):
                epoch_set = set()
                for _ in range(int(files * records_per_file / batches)):
                    op_list = self.evaluate(yield_op)
                    self.assertTrue(len(op_list) is batches)
                    for r in op_list:
                        self.assertTrue(r[0] not in epoch_set)
                        epoch_set.add(r[0])

    def testInvalidParams(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session():
            with self.assertRaises(errors_impl.InvalidArgumentError):
                self.evaluate(data_flow_ops.gen_data_flow_ops.record_input(file_pattern='nan', file_buffer_size=-90, file_parallelism=-438, file_shuffle_shift_ratio=-784, batch_size=-933, file_random_seed=-678, compression_type='nan'))
if __name__ == '__main__':
    test.main()