"""Tests for ensure_shape_op."""
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.platform import test

class EnsureShapeOpTest(xla_test.XLATestCase):

    def testEnsureShape(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session() as sess:
            p = array_ops.placeholder(dtypes.int32)
            with self.test_scope():
                op = check_ops.ensure_shape(p, (None, 3))
            expected_out = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
            self.assertAllEqual(expected_out, sess.run(op, {p: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]}))

    def testInvalidEnsureShape(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session() as sess:
            p = array_ops.placeholder(dtypes.int32)
            with self.test_scope():
                op = check_ops.ensure_shape(p, (None, 3, 3))
            with self.assertRaisesRegex(errors_impl.InvalidArgumentError, 'is not compatible with expected shape'):
                sess.run(op, {p: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]})
if __name__ == '__main__':
    test.main()