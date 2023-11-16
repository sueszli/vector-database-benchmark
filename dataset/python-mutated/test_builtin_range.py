import unittest
from test_case_base import TestCaseBase
import paddle

def test_range_1(stop: int):
    if False:
        i = 10
        return i + 15
    return range(stop)

def test_range_2(start: int, stop: int):
    if False:
        for i in range(10):
            print('nop')
    return range(start, stop)

def test_range_3(start: int, stop: int, step: int):
    if False:
        while True:
            i = 10
    return range(start, stop, step)

def test_range_4(stop: int, index: int):
    if False:
        for i in range(10):
            print('nop')
    return range(stop)[index]

def test_range_5(stop: int):
    if False:
        i = 10
        return i + 15
    return list(range(stop))

def test_range_6(stop: int, index: int):
    if False:
        print('Hello World!')
    return list(range(stop))[index]

def test_range_7(index: int, tensor: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    return list(range(len(tensor.shape)))[index]

def test_range_8(stop: int):
    if False:
        for i in range(10):
            print('nop')
    sum = 0
    for i in range(stop):
        sum += i
    return sum

def test_range_9(stop: int, tensor: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    for i in range(stop):
        tensor += i
    return tensor

def test_range_10(stop: int, tensor: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    for i in range(stop):
        for j in range(stop + 1):
            tensor += j
    return tensor

class TestExecutor(TestCaseBase):

    def test_cases(self):
        if False:
            return 10
        start = 3
        stop = 10
        step = 2
        index = 1
        tensor = paddle.randn((10, 10))
        self.assert_results(test_range_1, stop)
        self.assert_results(test_range_2, start, stop)
        self.assert_results(test_range_3, start, stop, step)
        self.assert_results(test_range_4, stop, index)
        self.assert_results(test_range_5, stop)
        self.assert_results(test_range_6, stop, index)
        self.assert_results(test_range_7, index, tensor)
        self.assert_results(test_range_8, stop)
        self.assert_results(test_range_9, stop, paddle.randn((10,)))
        self.assert_results(test_range_10, stop, paddle.randn((10,)))
if __name__ == '__main__':
    unittest.main()