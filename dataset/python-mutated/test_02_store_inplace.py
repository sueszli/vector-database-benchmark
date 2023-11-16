import unittest
from test_case_base import TestCaseBase
import paddle

def foo(x: int, y: paddle.Tensor):
    if False:
        return 10
    x = x + 1
    y = y + 1
    x += y
    return x

class TestStoreInplace(TestCaseBase):

    def test_simple(self):
        if False:
            return 10
        self.assert_results(foo, 1, paddle.to_tensor(2))
if __name__ == '__main__':
    unittest.main()