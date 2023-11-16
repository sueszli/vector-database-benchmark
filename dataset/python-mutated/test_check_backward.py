import pytest
import chainerx

def _check_backward_unary(fprop):
    if False:
        for i in range(10):
            print('nop')
    x = chainerx.array([1, 2, 1], chainerx.float32)
    x.require_grad()
    chainerx.check_backward(fprop, (x,), (chainerx.array([0, -2, 1], chainerx.float32),), (chainerx.full((3,), 0.001, chainerx.float32),))

def test_correct_backward_unary():
    if False:
        while True:
            i = 10
    _check_backward_unary(lambda xs: (xs[0] * xs[0],))

def test_incorrect_backward_unary():
    if False:
        print('Hello World!')

    def fprop(xs):
        if False:
            return 10
        (x,) = xs
        return ((x * x).as_grad_stopped() + x,)
    with pytest.raises(chainerx.GradientCheckError):
        _check_backward_unary(fprop)

def _check_backward_binary(fprop):
    if False:
        while True:
            i = 10
    chainerx.check_backward(fprop, (chainerx.array([1, -2, 1], chainerx.float32).require_grad(), chainerx.array([0, 1, 2], chainerx.float32).require_grad()), (chainerx.array([1, -2, 3], chainerx.float32),), (chainerx.full((3,), 0.001, chainerx.float32), chainerx.full((3,), 0.001, chainerx.float32)))

def test_correct_backward_binary():
    if False:
        return 10
    _check_backward_binary(lambda xs: (xs[0] * xs[1],))

def test_incorrect_backward_binary():
    if False:
        for i in range(10):
            print('nop')

    def fprop(xs):
        if False:
            for i in range(10):
                print('nop')
        (x, y) = xs
        return ((x * y).as_grad_stopped() + x + y,)
    with pytest.raises(chainerx.GradientCheckError):
        _check_backward_binary(fprop)

def test_correct_double_backward_unary():
    if False:
        for i in range(10):
            print('nop')
    chainerx.check_double_backward(lambda xs: (xs[0] * xs[0],), (chainerx.array([1, 2, 3], chainerx.float32).require_grad(),), (chainerx.ones((3,), chainerx.float32).require_grad(),), (chainerx.ones((3,), chainerx.float32),), (chainerx.full((3,), 0.001, chainerx.float32), chainerx.full((3,), 0.001, chainerx.float32)), 0.0001, 0.001)

def test_correct_double_backward_binary():
    if False:
        return 10
    chainerx.check_double_backward(lambda xs: (xs[0] * xs[1],), (chainerx.array([1, 2, 3], chainerx.float32).require_grad(), chainerx.ones((3,), chainerx.float32).require_grad()), (chainerx.ones((3,), chainerx.float32).require_grad(),), (chainerx.ones((3,), chainerx.float32), chainerx.ones((3,), chainerx.float32)), (chainerx.full((3,), 0.001, chainerx.float32), chainerx.full((3,), 0.001, chainerx.float32), chainerx.full((3,), 0.001, chainerx.float32)), 0.0001, 0.001)