"""Table-driven test for encode_proto op.

It tests that encode_proto is a lossless inverse of decode_proto
(for the specified fields).
"""
from absl.testing import parameterized
import numpy as np
from google.protobuf import text_format
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.kernel_tests.proto import proto_op_test_base as test_base
from tensorflow.python.kernel_tests.proto import test_example_pb2
from tensorflow.python.ops import array_ops

class EncodeProtoOpTestBase(test_base.ProtoOpTestBase, parameterized.TestCase):
    """Base class for testing proto encoding ops."""

    def __init__(self, decode_module, encode_module, methodName='runTest'):
        if False:
            i = 10
            return i + 15
        'EncodeProtoOpTestBase initializer.\n\n    Args:\n      decode_module: a module containing the `decode_proto_op` method\n      encode_module: a module containing  the `encode_proto_op` method\n      methodName: the name of the test method (same as for test.TestCase)\n    '
        super(EncodeProtoOpTestBase, self).__init__(methodName)
        self._decode_module = decode_module
        self._encode_module = encode_module

    def testBadSizesShape(self):
        if False:
            i = 10
            return i + 15
        if context.executing_eagerly():
            expected_error = (errors.InvalidArgumentError, 'Invalid shape for field double_value.')
        else:
            expected_error = (ValueError, 'Shape must be at least rank 2 but is rank 0')
        with self.assertRaisesRegexp(*expected_error):
            self.evaluate(self._encode_module.encode_proto(sizes=1, values=[np.double(1.0)], message_type='tensorflow.contrib.proto.TestValue', field_names=['double_value']))

    def testBadInputs(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesOpError('Unknown field: non_existent_field'):
            self.evaluate(self._encode_module.encode_proto(sizes=[[1]], values=[np.array([[0.0]], dtype=np.int32)], message_type='tensorflow.contrib.proto.TestValue', field_names=['non_existent_field']))
        with self.assertRaisesOpError('Incompatible type for field double_value.'):
            self.evaluate(self._encode_module.encode_proto(sizes=[[1]], values=[np.array([[0.0]], dtype=np.int32)], message_type='tensorflow.contrib.proto.TestValue', field_names=['double_value']))
        for sizes_value in (1, np.array([[[0, 0]]])):
            with self.assertRaisesOpError('sizes should be batch_size \\+ \\[len\\(field_names\\)\\]'):
                if context.executing_eagerly():
                    self.evaluate(self._encode_module.encode_proto(sizes=sizes_value, values=[np.array([[0.0]])], message_type='tensorflow.contrib.proto.TestValue', field_names=['double_value']))
                else:
                    with self.cached_session():
                        sizes = array_ops.placeholder(dtypes.int32)
                        values = array_ops.placeholder(dtypes.float64)
                        self._encode_module.encode_proto(sizes=sizes, values=[values], message_type='tensorflow.contrib.proto.TestValue', field_names=['double_value']).eval(feed_dict={sizes: sizes_value, values: [[0.0]]})
        with self.assertRaisesOpError('Values must match up to the last dimension'):
            if context.executing_eagerly():
                self.evaluate(self._encode_module.encode_proto(sizes=[[1, 1]], values=[np.array([[0.0]]), np.array([[0], [0]])], message_type='tensorflow.contrib.proto.TestValue', field_names=['double_value', 'int32_value']))
            else:
                with self.cached_session():
                    values1 = array_ops.placeholder(dtypes.float64)
                    values2 = array_ops.placeholder(dtypes.int32)
                    self._encode_module.encode_proto(sizes=[[1, 1]], values=[values1, values2], message_type='tensorflow.contrib.proto.TestValue', field_names=['double_value', 'int32_value']).eval(feed_dict={values1: [[0.0]], values2: [[0], [0]]})

    def _testRoundtrip(self, in_bufs, message_type, fields):
        if False:
            while True:
                i = 10
        field_names = [f.name for f in fields]
        out_types = [f.dtype for f in fields]
        with self.cached_session() as sess:
            (sizes, field_tensors) = self._decode_module.decode_proto(in_bufs, message_type=message_type, field_names=field_names, output_types=out_types)
            out_tensors = self._encode_module.encode_proto(sizes, field_tensors, message_type=message_type, field_names=field_names)
            (out_bufs,) = sess.run([out_tensors])
            self.assertEqual(in_bufs.shape, out_bufs.shape)
            for (in_buf, out_buf) in zip(in_bufs.flat, out_bufs.flat):
                in_obj = test_example_pb2.TestValue()
                in_obj.ParseFromString(in_buf)
                out_obj = test_example_pb2.TestValue()
                out_obj.ParseFromString(out_buf)
                self.assertEqual(in_obj, out_obj)
                self.assertEqual(in_buf, out_buf)

    @parameterized.named_parameters(*test_base.ProtoOpTestBase.named_parameters(extension=False))
    def testRoundtrip(self, case):
        if False:
            while True:
                i = 10
        in_bufs = [value.SerializeToString() for value in case.values]
        in_bufs = np.reshape(np.array(in_bufs, dtype=object), list(case.shapes))
        return self._testRoundtrip(in_bufs, 'tensorflow.contrib.proto.TestValue', case.fields)

    @parameterized.named_parameters(*test_base.ProtoOpTestBase.named_parameters(extension=False))
    def testRoundtripPacked(self, case):
        if False:
            for i in range(10):
                print('nop')
        in_bufs = [text_format.Parse(text_format.MessageToString(value, float_format='.17g'), test_example_pb2.PackedTestValue()).SerializeToString() for value in case.values]
        in_bufs = np.reshape(np.array(in_bufs, dtype=object), list(case.shapes))
        return self._testRoundtrip(in_bufs, 'tensorflow.contrib.proto.PackedTestValue', case.fields)