import unittest
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.base import core
np.random.seed(100)

def ref_poisson_nll_loss(input, label, log_input=True, full=False, epsilon=1e-08, reduction='mean'):
    if False:
        while True:
            i = 10
    if epsilon <= 0:
        raise ValueError('The value of `epsilon` in PoissonNLLLoss should be positve, but received %f, which is not allowed' % epsilon)
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError("The value of 'reduction' in SoftMarginLoss should be 'sum', 'mean' or 'none', but received %s, which is not allowed." % reduction)
    loss_out = 0
    if log_input:
        loss_out = np.exp(input) - label * input
    else:
        loss_out = input - label * np.log(input + epsilon)
    if full:
        stirling_approx = label * np.log(label) - label + 0.5 * np.log(2 * np.pi * label)
        loss_out += np.where(label > 1, stirling_approx, np.zeros_like(stirling_approx))
    if reduction == 'none':
        return loss_out
    elif reduction == 'sum':
        return [np.sum(loss_out)]
    elif reduction == 'mean':
        return [np.mean(loss_out)]

class TestPoissonNLLLossBasicCase(unittest.TestCase):

    def setUp(self, dtype='float32'):
        if False:
            while True:
                i = 10
        self.shape = [10, 2]
        self.dtype = dtype
        self.input_np = np.random.random(self.shape).astype(self.dtype)
        self.label_np = np.random.random(self.shape).astype(self.dtype)
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()

    def test_static_case(self, dtype='float32', log_input=True, full=False, epsilon=1e-08, reduction='mean'):
        if False:
            for i in range(10):
                print('nop')
        self.setUp(dtype)
        paddle.enable_static()
        prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(prog, startup_prog):
            input = paddle.static.data('input', self.shape, dtype)
            label = paddle.static.data('label', self.shape, dtype)
            input.desc.set_need_check_feed(False)
            label.desc.set_need_check_feed(False)
            out1 = F.poisson_nll_loss(input, label, log_input=log_input, full=full, epsilon=epsilon, reduction=reduction)
            poisson_nll_loss = paddle.nn.PoissonNLLLoss(log_input=log_input, full=full, epsilon=epsilon, reduction=reduction)
            out2 = poisson_nll_loss(input, label)
        exe = paddle.static.Executor(self.place)
        exe.run(startup_prog)
        res = exe.run(prog, feed={'input': self.input_np, 'label': self.label_np}, fetch_list=[out1, out2])
        out_ref = ref_poisson_nll_loss(self.input_np, self.label_np, log_input=log_input, full=full, epsilon=epsilon, reduction=reduction)
        for r in res:
            np.allclose(out_ref, r, rtol=1e-05)

    def test_dynamic_case(self, dtype='float32', log_input=True, full=False, epsilon=1e-08, reduction='mean', type=None):
        if False:
            print('Hello World!')
        self.setUp(dtype)
        paddle.disable_static(self.place)
        input_x = paddle.to_tensor(self.input_np)
        label = paddle.to_tensor(self.label_np)
        out_ref = ref_poisson_nll_loss(self.input_np, self.label_np, log_input=log_input, full=full, epsilon=epsilon, reduction=reduction)
        out1 = F.poisson_nll_loss(input_x, label, log_input=log_input, full=full, epsilon=epsilon, reduction=reduction)
        if type == 'test_err_reduction':
            self.assertRaises(ValueError, paddle.nn.functional.poisson_nll_loss, input=input_x, label=label, log_input=log_input, full=full, epsilon=epsilon, reduction='unsupport reduction')
        elif type == 'test_err_epsilon':
            self.assertRaises(ValueError, paddle.nn.functional.poisson_nll_loss, input=input_x, label=label, log_input=log_input, full=full, epsilon=-1, reduction='mean')
        poisson_nll_loss = paddle.nn.PoissonNLLLoss(log_input=log_input, full=full, epsilon=epsilon, reduction=reduction)
        out2 = poisson_nll_loss(input_x, label)
        for r in [out1, out2]:
            np.allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_api(self):
        if False:
            print('Hello World!')
        pass

class TestPoissonNLLLossErrCase(TestPoissonNLLLossBasicCase):

    def test_err_reduction(self):
        if False:
            while True:
                i = 10
        self.test_dynamic_case(type='test_err_reduction')

    def test_err_epsilon(self):
        if False:
            return 10
        self.test_dynamic_case(type='test_err_epsilon')

    def test_api(self):
        if False:
            print('Hello World!')
        self.test_err_reduction()
        self.test_err_epsilon()

class TestPoissonNLLLossFloat16Case(TestPoissonNLLLossBasicCase):

    def test_api(self):
        if False:
            while True:
                i = 10
        if core.is_compiled_with_cuda():
            self.test_static_case(dtype='float16')
            self.test_dynamic_case(dtype='float16')

class TestPoissonNLLLossBfloat16Case(TestPoissonNLLLossBasicCase):

    def test_api(self):
        if False:
            return 10
        if core.is_compiled_with_cuda():
            self.test_static_case(dtype='uint16')
            self.test_dynamic_case(dtype='uint16')

class TestPoissonNLLLossFloat32Case(TestPoissonNLLLossBasicCase):

    def test_api(self):
        if False:
            while True:
                i = 10
        self.test_static_case(dtype='float32')
        self.test_dynamic_case(dtype='float32')

class TestPoissonNLLLossFloat64Case(TestPoissonNLLLossBasicCase):

    def test_api(self):
        if False:
            return 10
        self.test_static_case(dtype='float64')
        self.test_dynamic_case(dtype='float64')

class TestPoissonNLLLossNoLoginputCase(TestPoissonNLLLossBasicCase):

    def test_api(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_static_case(log_input=False)
        self.test_dynamic_case(log_input=False)

class TestPoissonNLLLossFulllossCase(TestPoissonNLLLossBasicCase):

    def test_api(self):
        if False:
            while True:
                i = 10
        self.test_static_case(full=True)
        self.test_dynamic_case(full=True)

class TestPoissonNLLLossSumReductionCase(TestPoissonNLLLossBasicCase):

    def test_api(self):
        if False:
            return 10
        self.test_static_case(reduction='sum')
        self.test_dynamic_case(reduction='sum')
if __name__ == '__main__':
    unittest.main()