"""Test cases for operators with no arguments."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import googletest

class NullaryOpsTest(xla_test.XLATestCase):

    def _testNullary(self, op, expected):
        if False:
            return 10
        with self.session() as session:
            with self.test_scope():
                output = op()
            result = session.run(output)
            self.assertAllClose(result, expected, rtol=0.001)

    def testNoOp(self):
        if False:
            print('Hello World!')
        with self.session():
            with self.test_scope():
                output = control_flow_ops.no_op()
            output.run()

    def testConstants(self):
        if False:
            return 10
        for dtype in self.numeric_types:
            constants = [dtype(42), np.array([], dtype=dtype), np.array([1, 2], dtype=dtype), np.array([7, 7, 7, 7, 7], dtype=dtype), np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype), np.array([[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], dtype=dtype), np.array([[[]], [[]]], dtype=dtype), np.array([[[[1]]]], dtype=dtype)]
            for c in constants:
                self._testNullary(lambda c=c: constant_op.constant(c), expected=c)

    def testComplexConstants(self):
        if False:
            i = 10
            return i + 15
        for dtype in self.complex_types:
            constants = [dtype(42 + 3j), np.array([], dtype=dtype), np.ones([50], dtype=dtype) * (3 + 4j), np.array([1j, 2 + 1j], dtype=dtype), np.array([[1, 2j, 7j], [4, 5, 6]], dtype=dtype), np.array([[[1, 2], [3, 4 + 6j], [5, 6]], [[10 + 7j, 20], [30, 40], [50, 60]]], dtype=dtype), np.array([[[]], [[]]], dtype=dtype), np.array([[[[1 + 3j]]]], dtype=dtype)]
            for c in constants:
                self._testNullary(lambda c=c: constant_op.constant(c), expected=c)
if __name__ == '__main__':
    googletest.main()