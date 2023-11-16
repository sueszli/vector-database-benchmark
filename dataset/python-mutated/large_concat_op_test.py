"""Functional tests for Concat Op."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class LargeConcatOpTest(test.TestCase):
    """Tests that belong in concat_op_test.py, but run over large tensors."""

    def testConcatLargeTensors(self):
        if False:
            i = 10
            return i + 15
        with ops.device('/cpu:0'):
            a = array_ops.ones([2 ** 31 + 6], dtype=dtypes.int8)
            b = array_ops.zeros([1024], dtype=dtypes.int8)
            onezeros = array_ops.concat([a, b], 0)
        with self.session(use_gpu=False):
            _ = self.evaluate(onezeros)
if __name__ == '__main__':
    test.main()