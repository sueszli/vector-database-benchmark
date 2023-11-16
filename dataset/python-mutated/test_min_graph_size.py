from __future__ import annotations
import unittest
from test_case_base import TestCaseBase
import paddle
from paddle.jit import sot
from paddle.jit.sot.utils import min_graph_size_guard

def case_for(x, vars):
    if False:
        while True:
            i = 10
    x = x + 1
    sot.psdb.breakgraph()
    for y in vars:
        x += y
    return x

def case_if(x):
    if False:
        print('Hello World!')
    x = x + 1
    if x > 5:
        x += 3
    else:
        x += 4
    return x

def case_call(x):
    if False:
        i = 10
        return i + 15
    y = paddle.to_tensor(x.numpy())
    x += y
    return x

def call_with_kwargs_inner(x):
    if False:
        return 10
    return paddle.to_tensor(x.numpy())

def call_with_kwargs(x):
    if False:
        return 10
    y = call_with_kwargs_inner(x=x)
    x += y
    return x

def case_all(x, vars):
    if False:
        return 10
    x = x + 1
    for y in vars:
        z = paddle.to_tensor(x.numpy())
        x += z
        x += y
        if x > 5:
            x += y
        else:
            x += 3
    return x

class CustomLayer(paddle.nn.Layer):

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        return self.forward_features(x)

    def forward_features(self, x):
        if False:
            return 10
        return x.numpy()

class TestMinGraphSize(TestCaseBase):

    @min_graph_size_guard(10)
    def test_cases(self):
        if False:
            while True:
                i = 10
        x = paddle.to_tensor(1)
        self.assert_results(case_for, x, [1, 2, 3])
        self.assert_results(case_if, x)
        self.assert_results(case_call, x)
        self.assert_results(case_all, x, [4, 5, 6])

    @min_graph_size_guard(10)
    def test_layer(self):
        if False:
            return 10
        x = paddle.to_tensor(1)
        layer = CustomLayer()
        self.assert_results(layer.forward, x)

    @min_graph_size_guard(10)
    def test_call_with_kwargs(self):
        if False:
            while True:
                i = 10
        x = paddle.to_tensor(1)
        self.assert_results(call_with_kwargs, x)
if __name__ == '__main__':
    unittest.main()