"""Tests for quantized operations."""
import math
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest

class QuantizedOpsTest(xla_test.XLATestCase):

    def testQuantizedTypeRoundtrip(self):
        if False:
            return 10
        with self.session() as session:
            for dtype in self.quantized_tf_types:
                in_values = np.array([1, 2, 3, 4, 5, 6])
                expected = [[1, 2], [3, 4], [5, 6]]
                with self.test_scope():
                    p = array_ops.placeholder(dtype=dtypes.int32)
                    x = math_ops.cast(p, dtype)
                    x = array_ops.reshape(x, [3, 2])
                value = session.run(x, {p: in_values})
                self.assertAllEqual(value, expected)

class DequantizedOpsTest(xla_test.XLATestCase):

    def pack_uint8_r2_to_uint32(self, test_input):
        if False:
            return 10
        (num_rows, num_columns) = test_input.get_shape().as_list()
        num_output_columns = int(math.ceil(num_columns / 4.0))
        padding_input = array_ops.pad(math_ops.cast(test_input, dtype=dtypes.uint8), constant_op.constant([[0, 0], [0, num_output_columns * 4 - num_columns]]))
        output = array_ops.zeros([num_rows, num_output_columns], dtype=dtypes.uint32)
        num_elements_per_pack = 4
        shift_bits = 8
        iota_r1 = math_ops.range(num_output_columns * num_elements_per_pack)
        for p in range(num_elements_per_pack):
            selected_index = math_ops.equal(math_ops.mod(iota_r1, num_elements_per_pack), p)
            gather_index = array_ops.boolean_mask(iota_r1, selected_index)
            gathered_input = array_ops.gather(padding_input, gather_index, axis=1)
            total_shift_bits = shift_bits * (num_elements_per_pack - p - 1)
            left_shift_input = bitwise_ops.left_shift(math_ops.cast(gathered_input, dtype=dtypes.uint32), total_shift_bits)
            output = bitwise_ops.bitwise_or(output, left_shift_input)
        return output

    def testDequantizeQuint8(self):
        if False:
            print('Hello World!')
        num_rows = 100
        num_columns = 3547
        random_input = np.random.normal(128.0, 10.0, [num_rows, num_columns])
        with self.session() as session:
            with ops.device('CPU'):
                test_input = ops.convert_to_tensor(random_input, dtype=dtypes.float32)
                transposed_input = array_ops.transpose(test_input, [1, 0])
                quantized_input = array_ops.quantize(transposed_input, 0.0, 255.0, dtypes.quint8)
                packed_input = self.pack_uint8_r2_to_uint32(quantized_input.output)
            with self.test_scope():
                transposed_quantized_output = xla.dequantize(packed_input, 0.0, 255.0, 'MIN_COMBINED', True)
                quantized_output = array_ops.slice(transposed_quantized_output, [0, 0], [num_rows, num_columns])
            value = session.run(quantized_output)
        self.assertAllClose(value, random_input, 1.0)
if __name__ == '__main__':
    googletest.main()