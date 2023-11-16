"""Tests for PrecisionOp."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test

class InTopKTest(test.TestCase):

    def _validateInTopK(self, predictions, target, k, expected):
        if False:
            for i in range(10):
                print('nop')
        np_ans = np.array(expected, np.bool_)
        with self.cached_session():
            precision = nn_ops.in_top_k(predictions, target, k)
            out = self.evaluate(precision)
            self.assertAllClose(np_ans, out)
            self.assertShapeEqual(np_ans, precision)

    def testInTop1(self):
        if False:
            while True:
                i = 10
        predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
        target = [3, 2]
        self._validateInTopK(predictions, target, 1, [True, False])

    def testInTop2(self):
        if False:
            i = 10
            return i + 15
        predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
        target = [2, 2]
        self._validateInTopK(predictions, target, 2, [False, True])

    def testInTop2Tie(self):
        if False:
            return 10
        predictions = [[0.1, 0.3, 0.2, 0.2], [0.1, 0.3, 0.2, 0.2]]
        target = [2, 3]
        self._validateInTopK(predictions, target, 2, [True, True])

    def testInTop2_int64Target(self):
        if False:
            for i in range(10):
                print('nop')
        predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
        target = np.asarray([0, 2]).astype(np.int64)
        self._validateInTopK(predictions, target, 2, [False, True])

    def testInTopNan(self):
        if False:
            while True:
                i = 10
        predictions = [[0.1, float('nan'), 0.2, 0.4], [0.1, 0.2, 0.3, float('inf')]]
        target = [1, 3]
        self._validateInTopK(predictions, target, 2, [False, False])

    def testBadTarget(self):
        if False:
            print('Hello World!')
        predictions = [[0.1, 0.3, 0.2, 0.2], [0.1, 0.3, 0.2, 0.2]]
        target = [2, 4]
        self._validateInTopK(predictions, target, 2, [True, False])

    def testEmpty(self):
        if False:
            print('Hello World!')
        predictions = np.empty([0, 5])
        target = np.empty([0], np.int32)
        self._validateInTopK(predictions, target, 2, [])

    def testTensorK(self):
        if False:
            print('Hello World!')
        predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
        target = [0, 2]
        k = constant_op.constant(3)
        self._validateInTopK(predictions, target, k, [False, True])
if __name__ == '__main__':
    test.main()