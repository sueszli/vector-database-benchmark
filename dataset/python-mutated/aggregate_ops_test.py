"""Tests for aggregate_ops."""
import numpy as np
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test

class AddNTest(test.TestCase):
    _MAX_N = 10

    def _supported_types(self):
        if False:
            for i in range(10):
                print('nop')
        if test.is_gpu_available():
            return [dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128, dtypes.int64]
        return [dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.bfloat16, dtypes.float16, dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]

    def _buildData(self, shape, dtype):
        if False:
            while True:
                i = 10
        data = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
        if dtype.is_complex:
            return data + 10j * data
        return data

    def testAddN(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(12345)
        with self.session():
            for dtype in self._supported_types():
                for count in range(1, self._MAX_N + 1):
                    data = [self._buildData((2, 2), dtype) for _ in range(count)]
                    actual = self.evaluate(math_ops.add_n(data))
                    expected = np.sum(np.vstack([np.expand_dims(d, 0) for d in data]), axis=0)
                    self.assertAllCloseAccordingToType(expected, actual, float_rtol=5e-06, float_atol=5e-06, half_rtol=0.005, half_atol=0.005)

    @test_util.run_deprecated_v1
    def testUnknownShapes(self):
        if False:
            while True:
                i = 10
        np.random.seed(12345)
        with self.session() as sess:
            for dtype in self._supported_types():
                data = self._buildData((2, 2), dtype)
                for count in range(1, self._MAX_N + 1):
                    data_ph = array_ops.placeholder(dtype=dtype)
                    actual = sess.run(math_ops.add_n([data_ph] * count), {data_ph: data})
                    expected = np.sum(np.vstack([np.expand_dims(data, 0)] * count), axis=0)
                    self.assertAllCloseAccordingToType(expected, actual, half_rtol=0.005, half_atol=0.005)

    @test_util.run_deprecated_v1
    def testVariant(self):
        if False:
            i = 10
            return i + 15

        def create_constant_variant(value):
            if False:
                i = 10
                return i + 15
            return constant_op.constant(tensor_pb2.TensorProto(dtype=dtypes.variant.as_datatype_enum, tensor_shape=tensor_shape.TensorShape([]).as_proto(), variant_val=[tensor_pb2.VariantTensorDataProto(type_name=b'int', metadata=np.array(value, dtype=np.int32).tobytes())]))
        with self.session(use_gpu=False):
            num_tests = 127
            values = list(range(100))
            variant_consts = [create_constant_variant(x) for x in values]
            sum_count_indices = np.random.randint(1, 29, size=num_tests)
            sum_indices = [np.random.randint(100, size=count) for count in sum_count_indices]
            expected_sums = [np.sum(x) for x in sum_indices]
            variant_sums = [math_ops.add_n([variant_consts[i] for i in x]) for x in sum_indices]
            variant_sums_string = string_ops.as_string(variant_sums)
            self.assertAllEqual(variant_sums_string, ['Variant<type: int value: {}>'.format(s).encode('utf-8') for s in expected_sums])
if __name__ == '__main__':
    test.main()