"""Functional tests for slice op that consume a lot of GPU memory."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class SliceTest(test.TestCase):

    def testInt64Slicing(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session(force_gpu=test.is_gpu_available()):
            a_large = array_ops.tile(constant_op.constant(np.array([False, True] * 4)), [2 ** 29 + 3])
            slice_t = array_ops.slice(a_large, np.asarray([3]).astype(np.int64), [3])
            slice_val = self.evaluate(slice_t)
            self.assertAllEqual([True, False, True], slice_val)
            slice_t = array_ops.slice(a_large, constant_op.constant([long(2) ** 32 + 3], dtype=dtypes.int64), [3])
            slice_val = self.evaluate(slice_t)
            self.assertAllEqual([True, False, True], slice_val)