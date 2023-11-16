import unittest
import numpy as np
import paddle
from paddle import _pir_ops, nn
from paddle.autograd.ir_backward import grad
from paddle.decomposition import decompose
from paddle.framework import core
paddle.enable_static()

class SimpNet(nn.Layer):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

    def forward(self, x, linear1_weight, linear2_weight):
        if False:
            i = 10
            return i + 15
        x2 = _pir_ops.matmul(x, linear1_weight, False, False)
        x3 = _pir_ops.gelu(x2, False)
        res = _pir_ops.matmul(x3, linear2_weight, False, False)
        return res

class TestPrimMode(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        np.random.seed(2023)
        self.shape_x = [2, 1024, 1024]
        self.shape_y = [2, 1024, 1024]
        self.shape_l1_w = [2, 1024, 4096]
        self.shape_l2_w = [2, 4096, 1024]
        self.x = np.random.random(self.shape_x).astype('float32')
        self.y = np.random.random(self.shape_y).astype('float32')
        self.l1_w = np.random.random(self.shape_l1_w).astype('float32')
        self.l2_w = np.random.random(self.shape_l2_w).astype('float32')

    def base_net(self, flag=None):
        if False:
            print('Hello World!')
        if flag == 'backward':
            core._set_prim_backward_enabled(True)
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            net = SimpNet()
            x = paddle.static.data('x', self.shape_x, dtype='float32')
            y = paddle.static.data('y', self.shape_y, dtype='float32')
            x.stop_gradient = False
            y.stop_gradient = False
            l1_w = paddle.static.data('l1_w', self.shape_l1_w, dtype='float32')
            l2_w = paddle.static.data('l2_w', self.shape_l2_w, dtype='float32')
            divide_out = paddle.divide(x, y)
            res = net(divide_out, l1_w, l2_w)
            [res2] = decompose(main_program, [res])
            gradients = grad(res2, (x, y))
            if flag == 'backward':
                whole_ops_before = [op.name() for op in main_program.global_block().ops]
                assert 'pd_op.gelu' in whole_ops_before and 'pd_op.gelu_grad' not in whole_ops_before
                core._set_prim_forward_enabled(True)
                [res2] = decompose(main_program, [res2], whitelist={'pd_op.gelu'})
                whole_ops_after = [op.name() for op in main_program.global_block().ops]
                assert 'pd_op.gelu' not in whole_ops_after
                core._set_prim_forward_enabled(False)
            exe = paddle.static.Executor()
            outs = exe.run(feed={'x': self.x, 'y': self.y, 'l1_w': self.l1_w, 'l2_w': self.l2_w}, fetch_list=[res2, gradients[0], gradients[1]])
        if flag == 'backward':
            core._set_prim_backward_enabled(False)
        return outs

    def test_prim_custom_vjp(self):
        if False:
            return 10
        res_ref = self.base_net()
        res = self.base_net('backward')
        for (ref, actual) in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, rtol=1e-06)
if __name__ == '__main__':
    unittest.main()