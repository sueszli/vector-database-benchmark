"""Tests for where op."""
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu

class WhereOpTest(xla_test.XLATestCase):

    def __init__(self, method_name='runTest'):
        if False:
            i = 10
            return i + 15
        super(WhereOpTest, self).__init__(method_name)
        if config.list_logical_devices('TPU'):
            with self.session() as sess:
                sess.run(tpu.initialize_system())

    def testWhere(self):
        if False:
            while True:
                i = 10
        'Test first form of where (return indices).'
        with self.session() as sess:
            with self.test_scope():
                x = array_ops.placeholder(dtypes.bool)
                true_vals = array_ops.where(x)
            feed = [[True, False, False], [False, True, True]]
            self.assertAllEqual([[0, 0], [1, 1], [1, 2]], sess.run(true_vals, {x: feed}))

    def testWhereGather(self):
        if False:
            print('Hello World!')
        'Test where followed by a gather.'
        with self.session() as sess:
            with self.test_scope():
                x = array_ops.placeholder(dtypes.bool)
                value = array_ops.constant([[0, 1], [2, 3]], dtypes.float32)
                true_vals = array_ops.where(x)
                gathered = array_ops.gather_nd(value, true_vals)
            feed = [[True, False], [True, True]]
            self.assertAllEqual([0, 2, 3], sess.run(gathered, {x: feed}))

    def testWhereGatherReduce(self):
        if False:
            for i in range(10):
                print('nop')
        'Test where followed by a gather and a reduce.'
        with self.session() as sess:
            with self.test_scope():
                x = array_ops.placeholder(dtypes.bool)
                value = array_ops.constant([[0, 1], [2, 3]], dtypes.float32)
                indices = array_ops.where(x)
                gathered = array_ops.gather_nd(value, indices)
                reduction = math_ops.reduce_sum(gathered)
            feed = [[True, False], [True, True]]
            self.assertAllEqual(5, sess.run(reduction, {x: feed}))

    def testWhere1D(self):
        if False:
            i = 10
            return i + 15
        'Test first form of where (return indices).'
        with self.session() as sess:
            with self.test_scope():
                x = array_ops.placeholder(dtypes.bool)
                result = array_ops.where(x)
            feed = [True, False, True]
            self.assertAllEqual([[0], [2]], sess.run(result, {x: feed}))

    def testWhereInt(self):
        if False:
            i = 10
            return i + 15
        'Test Where with integers.'
        with self.session() as sess:
            with self.test_scope():
                x = array_ops.placeholder(dtypes.int32)
                result = array_ops.where(x)
            feed = [-1, 0, 1]
            self.assertAllEqual([[0], [2]], sess.run(result, {x: feed}))

    def testWhereFloat(self):
        if False:
            while True:
                i = 10
        'Test Where with floats.'
        with self.session() as sess:
            with self.test_scope():
                x = array_ops.placeholder(dtypes.float32)
                result = array_ops.where(x)
            feed = [-1.0, -0.0, 0.0, 1.0]
            self.assertAllEqual([[0], [3]], sess.run(result, {x: feed}))

    def testWhereComplex(self):
        if False:
            i = 10
            return i + 15
        'Test Where with floats.'
        with self.session() as sess:
            with self.test_scope():
                x = array_ops.placeholder(dtypes.complex64)
                result = array_ops.where(x)
            feed = [-1.0 + 0j, -0.0 + 0j, 0.0 - 0j, 1.0 - 1j, 1.0 + 0j, 0.0 + 1j]
            self.assertAllEqual([[0], [3], [4], [5]], sess.run(result, {x: feed}))
if __name__ == '__main__':
    test.main()