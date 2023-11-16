import unittest
from test_case_base import TestCaseBase
import paddle

def foo(x, y):
    if False:
        return 10
    ret = x + y
    return ret

class TestMultipleArgs(TestCaseBase):

    def test_multiple_args(self):
        if False:
            while True:
                i = 10
        x = paddle.to_tensor([1.0])
        y = paddle.to_tensor([2.0])
        self.assert_results(foo, x, y)
if __name__ == '__main__':
    unittest.main()