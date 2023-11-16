import unittest
from test_case_base import TestCaseBase
import paddle
from paddle.nn.functional import relu

def paddle_api_method_call(x: paddle.Tensor):
    if False:
        for i in range(10):
            print('nop')
    m = x + 2
    m = paddle.nn.functional.relu(m)
    return m

def paddle_api_function_call(x: paddle.Tensor):
    if False:
        for i in range(10):
            print('nop')
    m = x + 2
    m = relu(m)
    return m

def paddle_api_function_call_concat(x: paddle.Tensor, y: paddle.Tensor, axis: int):
    if False:
        return 10
    return paddle.concat([x, y], axis=axis)

class TestPaddleApiCall(TestCaseBase):

    def test_paddle_api_method_call(self):
        if False:
            i = 10
            return i + 15
        self.assert_results(paddle_api_method_call, paddle.to_tensor(2.0))
        self.assert_results(paddle_api_method_call, paddle.to_tensor(-5.0))
        self.assert_results(paddle_api_method_call, paddle.to_tensor(0.0))

    def test_paddle_api_function_call(self):
        if False:
            return 10
        self.assert_results(paddle_api_function_call, paddle.to_tensor(2.0))
        self.assert_results(paddle_api_function_call, paddle.to_tensor(-5.0))
        self.assert_results(paddle_api_function_call, paddle.to_tensor(0.0))

    def test_paddle_api_function_call_concat(self):
        if False:
            print('Hello World!')
        a = paddle.to_tensor([[1, 2], [3, 4]])
        b = paddle.to_tensor([[5, 6], [7, 8]])
        self.assert_results(paddle_api_function_call_concat, a, b, 0)
        self.assert_results(paddle_api_function_call_concat, a, b, 1)
if __name__ == '__main__':
    unittest.main()