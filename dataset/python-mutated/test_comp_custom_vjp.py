import unittest
import paddle
from paddle.base import core

class TestCustomVJP(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')

        def func():
            if False:
                print('Hello World!')
            x = paddle.rand((1,))
            x.stop_gradient = False
            return paddle.nn.functional.dropout(x)
        self.f = func
        self.ops_fwd_enable_bwd_disable = ('uniform_random', 'uniform_random', 'fill_constant', 'greater_equal', 'cast', 'elementwise_mul', 'scale', 'cast', 'fill_any_like', 'scale', 'elementwise_mul_grad')
        self.ops_fwd_disable_bwd_enable = ('uniform_random', 'dropout', 'fill_any_like', 'fill_any_like', 'cast', 'elementwise_mul', 'scale')
        self.ops_all_enable = ('uniform_random', 'uniform_random', 'fill_constant', 'greater_equal', 'cast', 'elementwise_mul', 'scale', 'cast', 'fill_constant', 'fill_constant', 'cast', 'elementwise_mul', 'scale')

    def test_enable_prim_fwd(self):
        if False:
            print('Hello World!')
        core._set_prim_forward_enabled(True)
        core._set_prim_backward_enabled(False)
        self.assertEqual(self.ops_fwd_enable_bwd_disable, tuple((op.type for op in paddle.jit.to_static(full_graph=True)(self.f).get_concrete_program()[1]._train_program.block(0).ops)))
        core._set_prim_forward_enabled(False)
        core._set_prim_backward_enabled(False)

    def test_enable_prim_bwd(self):
        if False:
            return 10
        core._set_prim_forward_enabled(False)
        core._set_prim_backward_enabled(True)
        self.assertEqual(self.ops_fwd_disable_bwd_enable, tuple((op.type for op in paddle.jit.to_static(full_graph=True)(self.f).get_concrete_program()[1]._train_program.block(0).ops)))
        core._set_prim_forward_enabled(False)
        core._set_prim_backward_enabled(False)

    def test_enable_prim_all(self):
        if False:
            i = 10
            return i + 15
        core._set_prim_all_enabled(True)
        self.assertEqual(self.ops_all_enable, tuple((op.type for op in paddle.jit.to_static(full_graph=True)(self.f).get_concrete_program()[1]._train_program.block(0).ops)))
        core._set_prim_all_enabled(False)
if __name__ == '__main__':
    unittest.main()