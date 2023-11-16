"""Tests tf.ragged.fill_empty_rows."""
import numpy as np
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest

class RaggedFillEmptyRowsTest(test_util.TensorFlowTestCase):

    def testFillInt(self):
        if False:
            print('Hello World!')
        with test_util.use_gpu():
            default_value = constant_op.constant(-1, dtype=dtypes.int32)
            ragged_input = ragged_factory_ops.constant([[], [1, 3, 5, 7], [], [2, 4, 6, 8], []], dtype=dtypes.int32)
            (ragged_output, empty_row_indicator) = ragged_array_ops.fill_empty_rows(ragged_input, default_value)
            self.assertAllEqual(ragged_output, [[-1], [1, 3, 5, 7], [-1], [2, 4, 6, 8], [-1]])
            self.assertAllEqual(ragged_output.row_lengths(), [1, 4, 1, 4, 1])
            self.assertAllEqual(ragged_output.values, [-1, 1, 3, 5, 7, -1, 2, 4, 6, 8, -1])
            self.assertAllEqual(empty_row_indicator, [True, False, True, False, True])

    @test_util.run_deprecated_v1
    def testFillFloat(self):
        if False:
            print('Hello World!')
        with self.session():
            values = constant_op.constant([1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0], dtype=dtypes.float64)
            default_value = constant_op.constant(-1.0, dtype=dtypes.float64)
            ragged_input = ragged_tensor.RaggedTensor.from_row_lengths(values=values, row_lengths=[0, 4, 0, 4, 0])
            (ragged_output, empty_row_indicator) = ragged_array_ops.fill_empty_rows(ragged_input, default_value)
            self.assertAllEqual(ragged_output.row_lengths(), [1, 4, 1, 4, 1])
            self.assertAllEqual(ragged_output.values, [-1.0, 1.0, 3.0, 5.0, 7.0, -1.0, 2.0, 4.0, 6.0, 8.0, -1.0])
            self.assertAllEqual(empty_row_indicator, [True, False, True, False, True])
            values_grad_err = gradient_checker.compute_gradient_error(values, values.shape.as_list(), ragged_output.values, [11], delta=1e-08)
            self.assertGreater(values_grad_err, 0)
            self.assertLess(values_grad_err, 1e-08)
            default_value_grad_err = gradient_checker.compute_gradient_error(default_value, default_value.shape.as_list(), ragged_output.values, [11], delta=1e-08)
            self.assertGreater(default_value_grad_err, 0)
            self.assertLess(default_value_grad_err, 1e-08)

    def testFillFloatFunction(self):
        if False:
            for i in range(10):
                print('nop')

        @def_function.function
        def func(ragged_input):
            if False:
                while True:
                    i = 10
            default_value = constant_op.constant(-1.0, dtype=dtypes.float64)
            return ragged_array_ops.fill_empty_rows(ragged_input, default_value)
        ragged_input = ragged_factory_ops.constant([[], [1.0, 3.0, 5.0, 7.0], [], [2.0, 4.0, 6.0, 8.0], []], dtype=dtypes.float64)
        (ragged_output, empty_row_indicator) = func(ragged_input)
        self.assertAllEqual(ragged_output.row_lengths(), [1, 4, 1, 4, 1])
        self.assertAllEqual(ragged_output.values, [-1.0, 1.0, 3.0, 5.0, 7.0, -1.0, 2.0, 4.0, 6.0, 8.0, -1.0])
        self.assertAllEqual(empty_row_indicator, [True, False, True, False, True])

    def testFillString(self):
        if False:
            i = 10
            return i + 15
        with test_util.force_cpu():
            values = constant_op.constant(['a', 'c', 'e', 'g', 'b', 'd', 'f', 'h'], dtype=dtypes.string)
            default_value = constant_op.constant('x', dtype=dtypes.string)
            ragged_input = ragged_tensor.RaggedTensor.from_row_lengths(values=values, row_lengths=[0, 4, 0, 4, 0])
            (ragged_output, empty_row_indicator) = ragged_array_ops.fill_empty_rows(ragged_input, default_value)
            self.assertAllEqual(ragged_output.row_lengths(), [1, 4, 1, 4, 1])
            self.assertAllEqual(ragged_output.values, [b'x', b'a', b'c', b'e', b'g', b'x', b'b', b'd', b'f', b'h', b'x'])
            self.assertAllEqual(empty_row_indicator, np.array([1, 0, 1, 0, 1]).astype(np.bool_))

    def testNoEmptyRows(self):
        if False:
            i = 10
            return i + 15
        with test_util.use_gpu():
            values = constant_op.constant([1, 3, 5, 7, 2, 4, 6, 8], dtype=dtypes.int32)
            default_value = constant_op.constant(-1, dtype=dtypes.int32)
            ragged_input = ragged_tensor.RaggedTensor.from_row_lengths(values=values, row_lengths=[4, 4])
            (ragged_output, empty_row_indicator) = ragged_array_ops.fill_empty_rows(ragged_input, default_value)
            self.assertAllEqual(ragged_output.row_lengths(), [4, 4])
            self.assertAllEqual(ragged_output.values, values)
            self.assertAllEqual(empty_row_indicator, np.zeros(2).astype(np.bool_))
if __name__ == '__main__':
    googletest.main()