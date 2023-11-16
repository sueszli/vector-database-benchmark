import unittest
from test_case_base import TestCaseBase
import paddle

def func_1(format_str, tensor):
    if False:
        i = 10
        return i + 15
    str = format_str.format(xx=12)
    a = '{xx} = 12'.format
    ttt = f'{10} = 12'
    a(xx=12)
    tensor = tensor + 1
    return (str, tensor)

def func_2(format_str, tensor):
    if False:
        i = 10
        return i + 15
    str = format_str % 10
    tensor = tensor + 1
    return (str, tensor)

class TestConstantGraph(TestCaseBase):

    def test_case_1(self):
        if False:
            return 10
        x = '{xx} is xx'
        tensor = paddle.to_tensor(1)
        self.assert_results(func_1, x, tensor)

    def test_case_2(self):
        if False:
            i = 10
            return i + 15
        x = '%s is xx'
        tensor = paddle.to_tensor(1)
        self.assert_results(func_2, x, tensor)
if __name__ == '__main__':
    unittest.main()