import unittest
import numpy as np
import paddle
from paddle.autograd.ir_backward import grad
from paddle.decomposition import decompose
from paddle.framework import core
paddle.enable_static()

class TestPrimMode(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        np.random.seed(2023)
        self.shape_x = [8, 16, 32, 64]
        self.shape_y = [8, 16, 32, 64]
        self.x = np.random.random(self.shape_x).astype('float32')
        self.y = np.random.random(self.shape_y).astype('float32')

    def base_net(self, flag=None):
        if False:
            print('Hello World!')
        if flag == 'forward':
            core._set_prim_forward_enabled(True)
        elif flag == 'backward':
            core._set_prim_backward_enabled(True)
        elif flag == 'all':
            core._set_prim_all_enabled(True)
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data('x', self.shape_x, dtype='float32')
            y = paddle.static.data('y', self.shape_y, dtype='float32')
            x.stop_gradient = False
            y.stop_gradient = False
            divide_out = paddle.divide(x, y)
            sum_out = paddle.mean(divide_out, axis=0)
            [new_out] = decompose(main_program, [sum_out])
            gradients = grad(new_out, (x, y))
            exe = paddle.static.Executor()
            [fwd, dx, dy] = exe.run(feed={'x': self.x, 'y': self.y}, fetch_list=[new_out, gradients])
        whole_ops = [op.name() for op in main_program.global_block().ops]
        if flag == 'forward':
            core._set_prim_forward_enabled(False)
            assert 'pd_op.mean' not in whole_ops and 'pd_op.divide_grad' in whole_ops
        elif flag == 'backward':
            core._set_prim_backward_enabled(False)
            assert 'pd_op.mean' in whole_ops and 'pd_op.divide_grad' not in whole_ops
        elif flag == 'all':
            core._set_prim_all_enabled(False)
            assert 'pd_op.mean' not in whole_ops and 'pd_op.divide_grad' not in whole_ops
        else:
            assert 'pd_op.mean' in whole_ops and 'pd_op.divide_grad' in whole_ops
        return (fwd, dx, dy)

    def test_prim_forward(self):
        if False:
            for i in range(10):
                print('nop')
        res_ref = self.base_net()
        res = self.base_net('forward')
        for (ref, actual) in zip(res_ref, res):
            np.testing.assert_equal(ref, actual)

    def test_prim_backward(self):
        if False:
            print('Hello World!')
        res_ref = self.base_net()
        res = self.base_net('backward')
        for (ref, actual) in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, rtol=1e-06)

    def test_prim_all(self):
        if False:
            while True:
                i = 10
        res_ref = self.base_net()
        res = self.base_net('all')
        for (ref, actual) in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, rtol=1e-06)
if __name__ == '__main__':
    unittest.main()