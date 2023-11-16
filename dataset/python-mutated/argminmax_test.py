"""Functional tests for ArgMin and ArgMax Ops."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class ArgMinMaxTest(xla_test.XLATestCase):

    def _assertOpOutputMatchesExpected(self, op, axis, output_type, op_input, expected):
        if False:
            i = 10
            return i + 15
        "Verifies that 'op' produces 'expected' when fed input 'op_input' .\n\n    Args:\n      op: argmin or argmax operator to test.\n      axis: integer axis to reduce across.\n      output_type: numpy datatype of the output to produce.\n      op_input: numpy input array to use as input to 'op'.\n      expected: numpy array representing the expected output of 'op'.\n    "
        with self.session() as session:
            with self.test_scope():
                pinp = array_ops.placeholder(dtypes.as_dtype(op_input.dtype), op_input.shape, name='a')
                output = op(pinp, axis=axis, output_type=output_type)
            result = session.run(output, {pinp: op_input})
            self.assertAllEqual(result, expected)

    def testArgMinMax(self):
        if False:
            print('Hello World!')
        minmax_types = self.all_types & {np.int32, np.int64}
        for dtype in self.int_types | self.float_types:
            for output_type in minmax_types:
                self._assertOpOutputMatchesExpected(math_ops.argmax, axis=0, output_type=output_type, op_input=np.array([1, 10, 27, 3, 3, 4], dtype=dtype), expected=output_type(2))
                self._assertOpOutputMatchesExpected(math_ops.argmax, axis=0, output_type=output_type, op_input=np.array([[4, 1, 7], [3, 2, 4]], dtype=dtype), expected=np.array([0, 1, 0], dtype=output_type))
                self._assertOpOutputMatchesExpected(math_ops.argmax, axis=1, output_type=output_type, op_input=np.array([[4, 1], [3, 2]], dtype=dtype), expected=np.array([0, 0], dtype=output_type))
                self._assertOpOutputMatchesExpected(math_ops.argmin, axis=0, output_type=output_type, op_input=np.array([3, 10, 27, 3, 2, 4], dtype=dtype), expected=output_type(4))
                self._assertOpOutputMatchesExpected(math_ops.argmin, axis=0, output_type=output_type, op_input=np.array([[4, 1, 7], [3, 2, 4]], dtype=dtype), expected=np.array([1, 0, 1], dtype=output_type))
                self._assertOpOutputMatchesExpected(math_ops.argmin, axis=1, output_type=output_type, op_input=np.array([[4, 1], [3, 2]], dtype=dtype), expected=np.array([1, 1], dtype=output_type))
if __name__ == '__main__':
    test.main()