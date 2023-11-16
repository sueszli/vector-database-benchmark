"""Tests for autograph_ops module."""
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import autograph_ops
from tensorflow.python.platform import test

class AutographOpsTest(test.TestCase):

    def test_wrap_py_func_dummy_return(self):
        if False:
            print('Hello World!')
        side_counter = [0]

        def test_fn(_):
            if False:
                return 10
            side_counter[0] += 1
        with self.cached_session():
            result = autograph_ops.wrap_py_func(test_fn, (5,))
            self.assertEqual(1, self.evaluate(result))
            self.assertEqual([1], side_counter)
            result = autograph_ops.wrap_py_func(test_fn, (constant_op.constant(5),))
            self.assertEqual(1, self.evaluate(result))
            self.assertEqual([2], side_counter)
if __name__ == '__main__':
    test.main()