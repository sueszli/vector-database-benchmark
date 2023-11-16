import unittest
from test_case_base import TestCaseBase
import paddle
from paddle.jit.sot.utils import strict_mode_guard

def test_enumerate_1(x: int, y: int):
    if False:
        while True:
            i = 10
    for (id, val) in enumerate(range(x)):
        if id % 2 == 0:
            y += val
    return y

def test_enumerate_2(x: list):
    if False:
        i = 10
        return i + 15
    return list(enumerate(x))

def test_enumerate_3(x: list):
    if False:
        while True:
            i = 10
    return tuple(enumerate(x))

def test_enumerate_4(x: paddle.Tensor):
    if False:
        for i in range(10):
            print('nop')
    sum = 0
    for (idx, val) in enumerate(x):
        sum += val
    return sum

def test_enumerate_5(x: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    sum = 0
    for (idx, val) in enumerate(x):
        for i in range(val):
            sum += val
    return sum

def test_enumerate_6(x: paddle.Tensor):
    if False:
        for i in range(10):
            print('nop')
    sum = 0
    for (idx, val) in enumerate(x):
        for i in range(idx):
            sum += val
    return sum

def test_enumerate_7(x: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    sum = 0
    x = x.flatten()
    for (idx, val) in enumerate(x):
        sum += val
    return sum

def test_enumerate_8(x: paddle.Tensor):
    if False:
        return 10
    sum = 0
    x = paddle.nonzero(x, as_tuple=False)
    for (idx, val) in enumerate(x):
        sum += val
    return sum

def test_enumerate_10(layer_list, x):
    if False:
        while True:
            i = 10
    sum = 0
    for (idx, layer) in enumerate(layer_list):
        sum += layer(x)
    return sum

class TestExecutor(TestCaseBase):

    def test_cases(self):
        if False:
            print('Hello World!')
        x = 8
        y = 5
        ty = paddle.randn((10, 10))
        layer_list = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for _ in range(3)])
        self.assert_results(test_enumerate_1, x, y)
        self.assert_results(test_enumerate_2, [2, 4, 6, 8, 10])
        self.assert_results(test_enumerate_3, [2, 4, 6, 8, 10])
        self.assert_results(test_enumerate_4, ty)
        with strict_mode_guard(False):
            self.assert_results(test_enumerate_5, paddle.to_tensor([1, 2, 3]))
        self.assert_results(test_enumerate_6, paddle.to_tensor([1, 2, 3]))
        self.assert_results(test_enumerate_7, ty)
        with strict_mode_guard(False):
            self.assert_results(test_enumerate_8, ty)
        self.assert_results(test_enumerate_10, layer_list, paddle.randn((10,)))
if __name__ == '__main__':
    unittest.main()