"""Tests for tensorflow.python.ops.op_def_library."""
from absl.testing import parameterized
import numpy as np
from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import _op_def_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

class OpDefUtilTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    @parameterized.parameters([('any', 'Foo', 'Foo'), ('any', 12, 12), ('any', {2: 3}, {2: 3}), ('string', 'Foo', 'Foo'), ('string', b'Foo', b'Foo'), ('int', 12, 12), ('int', 12.3, 12), ('float', 12, 12.0), ('float', 12.3, 12.3), ('bool', True, True), ('shape', tensor_shape.TensorShape([3]), tensor_shape.TensorShape([3])), ('shape', [3], tensor_shape.TensorShape([3])), ('type', dtypes.int32, dtypes.int32), ('type', np.int32, dtypes.int32), ('type', 'int32', dtypes.int32), ('tensor', tensor_pb2.TensorProto(dtype=types_pb2.DataType.DT_FLOAT), tensor_pb2.TensorProto(dtype=types_pb2.DataType.DT_FLOAT)), ('tensor', 'dtype: DT_FLOAT', tensor_pb2.TensorProto(dtype=types_pb2.DataType.DT_FLOAT)), ('list(any)', [1, 'foo', 7.3, dtypes.int32], [1, 'foo', 7.3, dtypes.int32]), ('list(any)', (1, 'foo'), [1, 'foo']), ('list(string)', ['foo', 'bar'], ['foo', 'bar']), ('list(string)', ('foo', 'bar'), ['foo', 'bar']), ('list(string)', iter('abcd'), ['a', 'b', 'c', 'd']), ('list(int)', (1, 2.3), [1, 2]), ('list(float)', (1, 2.3), [1.0, 2.3]), ('list(bool)', [True, False], [True, False]), ('list(type)', [dtypes.int32, dtypes.bool], [dtypes.int32, dtypes.bool]), ('list(shape)', [tensor_shape.TensorShape([3]), [4, 5]], [tensor_shape.TensorShape([3]), tensor_shape.TensorShape([4, 5])]), ('list(tensor)', [tensor_pb2.TensorProto(dtype=types_pb2.DataType.DT_FLOAT), 'dtype: DT_INT32'], [tensor_pb2.TensorProto(dtype=types_pb2.DataType.DT_FLOAT), tensor_pb2.TensorProto(dtype=types_pb2.DataType.DT_INT32)])])
    def testConvert(self, attr_type, value, expected):
        if False:
            for i in range(10):
                print('nop')
        result = _op_def_util.ConvertPyObjectToAttributeType(value, attr_type)
        self.assertEqual(expected, result)
        self.assertEqual(type(expected), type(result))
        if isinstance(result, list):
            for (expected_item, result_item) in zip(expected, result):
                self.assertEqual(type(expected_item), type(result_item))

    @parameterized.parameters([('string', 12), ('int', 'foo'), ('float', 'foo'), ('bool', 1), ('dtype', None), ('shape', 12.0), ('tensor', [1, 2, 3]), ('list(any)', 12), ('list(int)', [1, 'two']), ('list(string)', [1, 'two']), ('tensor', 'string that is not a text-formatted TensorProto')])
    def testConvertError(self, attr_type, value):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(TypeError, 'Failed to convert value'):
            _op_def_util.ConvertPyObjectToAttributeType(value, attr_type)

    @parameterized.parameters([("s: 'foo'", 'foo'), ('i: 5', 5), ('f: 8', 8.0), ('b: True', True), ('type: DT_INT32', dtypes.int32), ('shape { dim: [{size: 3}, {size: 4}] }', tensor_shape.TensorShape([3, 4])), ('list { }', []), ('list { s: [] }', []), ("list { s: ['a', 'b', 'c'] }", ['a', 'b', 'c']), ('list { i: [1, 2, 3] }', [1, 2, 3]), ('list { f: [2.0, 4.0] }', [2.0, 4.0])])
    def testAttrValueToPyObject(self, pbtxt, expected):
        if False:
            i = 10
            return i + 15
        proto = attr_value_pb2.AttrValue()
        text_format.Parse(pbtxt, proto)
        result = _op_def_util.SerializedAttrValueToPyObject(proto.SerializeToString())
        self.assertEqual(expected, result)

    @parameterized.parameters(['', 'tensor {}', 'func {}', "placeholder: ''", 'list { tensor [{}] }', 'list { func [{}] }'])
    def testAttrValueToPyObjectError(self, pbtxt):
        if False:
            print('Hello World!')
        proto = attr_value_pb2.AttrValue()
        text_format.Parse(pbtxt, proto)
        with self.assertRaises((TypeError, ValueError)):
            _op_def_util.SerializedAttrValueToPyObject(proto.SerializeToString())
if __name__ == '__main__':
    googletest.main()