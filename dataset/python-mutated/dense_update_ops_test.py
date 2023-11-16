"""Tests for tensorflow.ops.tf.Assign*."""
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

class AssignOpTest(test.TestCase):

    def _initAssignFetch(self, x, y, use_gpu):
        if False:
            i = 10
            return i + 15
        'Initialize a param to init and update it with y.'
        super(AssignOpTest, self).setUp()
        with test_util.device(use_gpu=use_gpu):
            p = variables.Variable(x)
            assign = state_ops.assign(p, y)
            self.evaluate(p.initializer)
            new_value = self.evaluate(assign)
            return (self.evaluate(p), new_value)

    def _initAssignAddFetch(self, x, y, use_gpu):
        if False:
            print('Hello World!')
        'Initialize a param to init, and compute param += y.'
        with test_util.device(use_gpu=use_gpu):
            p = variables.Variable(x)
            add = state_ops.assign_add(p, y)
            self.evaluate(p.initializer)
            new_value = self.evaluate(add)
            return (self.evaluate(p), new_value)

    def _initAssignSubFetch(self, x, y, use_gpu):
        if False:
            while True:
                i = 10
        'Initialize a param to init, and compute param -= y.'
        with test_util.device(use_gpu=use_gpu):
            p = variables.Variable(x)
            sub = state_ops.assign_sub(p, y)
            self.evaluate(p.initializer)
            new_value = self.evaluate(sub)
            return (self.evaluate(p), new_value)

    def _testTypes(self, vals):
        if False:
            print('Hello World!')
        for dtype in [np.float32, np.float64, np.int32, np.int64, dtypes.bfloat16.as_numpy_dtype]:
            x = np.zeros(vals.shape).astype(dtype)
            y = vals.astype(dtype)
            (var_value, op_value) = self._initAssignFetch(x, y, use_gpu=False)
            self.assertAllEqual(y, var_value)
            self.assertAllEqual(y, op_value)
            (var_value, op_value) = self._initAssignAddFetch(x, y, use_gpu=False)
            self.assertAllEqual(x + y, var_value)
            self.assertAllEqual(x + y, op_value)
            (var_value, op_value) = self._initAssignSubFetch(x, y, use_gpu=False)
            self.assertAllEqual(x - y, var_value)
            self.assertAllEqual(x - y, op_value)
            if test.is_built_with_gpu_support() and dtype in [np.float32, np.float64]:
                (var_value, op_value) = self._initAssignFetch(x, y, use_gpu=True)
                self.assertAllEqual(y, var_value)
                self.assertAllEqual(y, op_value)
                (var_value, op_value) = self._initAssignAddFetch(x, y, use_gpu=True)
                self.assertAllEqual(x + y, var_value)
                self.assertAllEqual(x + y, op_value)
                (var_value, op_value) = self._initAssignSubFetch(x, y, use_gpu=True)
                self.assertAllEqual(x - y, var_value)
                self.assertAllEqual(x - y, op_value)

    def testBasic(self):
        if False:
            while True:
                i = 10
        self._testTypes(np.arange(0, 20).reshape([4, 5]))
if __name__ == '__main__':
    test.main()