import numpy as np
import pytest
import megengine as mge
from megengine.amp import GradScaler
from megengine.autodiff import GradManager
from megengine.jit import trace

@pytest.mark.parametrize('is_trace', [False, True])
def test_grad_scaler(is_trace):
    if False:
        while True:
            i = 10
    gm = GradManager()
    scaler = GradScaler()

    def f(idx, data, calc):
        if False:
            while True:
                i = 10
        x = mge.tensor(data, no_cache=True)
        y = mge.tensor(data, no_cache=True)
        if is_trace:
            calc = trace(calc)
        gm.attach([x, y])
        with gm:
            loss = calc(x, y)
            scaler.backward(gm, loss, unscale_grad=False)
        np.testing.assert_equal(x.grad.numpy(), 2 * scaler.scale_factor)
        scaler.unscale(filter(lambda t: t.grad is not None, gm.attached_tensors()))
        np.testing.assert_equal(x.grad.numpy(), 2)

    def double_variables(x, y):
        if False:
            i = 10
            return i + 15
        z = x + 2 * y
        loss = 2 * z + 1
        return loss

    def single_variable(x, y):
        if False:
            for i in range(10):
                print('nop')
        z = x + 1
        loss = 2 * z + 1
        return loss

    def double_variables_with_same_grad(x, y):
        if False:
            return 10
        z = x + y
        loss = 2 * z + 1
        return loss
    for data in [np.random.random((1, 2, 3, 4)), 1.0]:
        for calc in [double_variables, single_variable, double_variables_with_same_grad]:
            for idx in range(3):
                f(idx, data, calc)