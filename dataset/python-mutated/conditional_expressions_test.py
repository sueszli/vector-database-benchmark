"""Tests for conditional_expressions module."""
from tensorflow.python.autograph.operators import conditional_expressions
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

def _basic_expr(cond):
    if False:
        while True:
            i = 10
    return conditional_expressions.if_exp(cond, lambda : constant_op.constant(1), lambda : constant_op.constant(2), 'cond')

@test_util.run_all_in_graph_and_eager_modes
class IfExpTest(test.TestCase):

    def test_tensor(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.evaluate(_basic_expr(constant_op.constant(True))), 1)
        self.assertEqual(self.evaluate(_basic_expr(constant_op.constant(False))), 2)

    def test_tensor_mismatched_type(self):
        if False:
            for i in range(10):
                print('nop')

        @def_function.function
        def test_fn():
            if False:
                while True:
                    i = 10
            conditional_expressions.if_exp(constant_op.constant(True), lambda : 1.0, lambda : 2, 'expr_repr')
        with self.assertRaisesRegex(TypeError, "'expr_repr' has dtype float32 in the main.*int32 in the else"):
            test_fn()

    def test_python(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.evaluate(_basic_expr(True)), 1)
        self.assertEqual(self.evaluate(_basic_expr(False)), 2)
        self.assertEqual(conditional_expressions.if_exp(True, lambda : 1, lambda : 2, ''), 1)
        self.assertEqual(conditional_expressions.if_exp(False, lambda : 1, lambda : 2, ''), 2)
if __name__ == '__main__':
    test.main()