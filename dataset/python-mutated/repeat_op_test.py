"""Tests for tensorflow.ops.array_ops.repeat."""
from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class RepeatTest(xla_test.XLATestCase):

    def test(self):
        if False:
            i = 10
            return i + 15

        @def_function.function(jit_compile=True)
        def repeat(values, repeats, axis):
            if False:
                print('Hello World!')
            return array_ops.repeat(values, repeats, axis)
        with self.session() as sess:
            with self.test_scope():
                values = array_ops.constant([[1, 2], [3, 4]], dtype=dtypes.int32)
                repeats = array_ops.constant([1, 2], dtype=dtypes.int32)
                y1 = repeat(values, repeats, 0)
                y2 = repeat(values, repeats, 1)
            (actual1, actual2) = sess.run([y1, y2])
        self.assertAllEqual(actual1, [[1, 2], [3, 4], [3, 4]])
        self.assertAllEqual(actual2, [[1, 2, 2], [3, 4, 4]])
if __name__ == '__main__':
    test.main()