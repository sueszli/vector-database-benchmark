from __future__ import annotations
import unittest
from test_case_base import TestCaseBase
import paddle
from paddle.jit.sot.psdb import check_no_breakgraph

@check_no_breakgraph
def pop_jump_if_false(x: bool, y: paddle.Tensor):
    if False:
        print('Hello World!')
    if x:
        y += 1
    else:
        y -= 1
    return y

@check_no_breakgraph
def pop_jump_if_true(x: bool, y: bool, z: paddle.Tensor):
    if False:
        for i in range(10):
            print('nop')
    return (x or y) and z

@check_no_breakgraph
def jump_if_false_or_pop(x: bool, y: paddle.Tensor):
    if False:
        return 10
    return x and y + 1

@check_no_breakgraph
def jump_if_true_or_pop(x: bool, y: paddle.Tensor):
    if False:
        while True:
            i = 10
    return x or y + 1

@check_no_breakgraph
def jump_absolute(x: int, y: paddle.Tensor):
    if False:
        return 10
    while x > 0:
        y += 1
        x -= 1
    return y

@check_no_breakgraph
def pop_jump_if_none(x: bool, y: paddle.Tensor):
    if False:
        return 10
    if x is not None:
        y += 1
    else:
        y -= 1
    return y

@check_no_breakgraph
def pop_jump_if_not_none(x: bool, y: paddle.Tensor):
    if False:
        return 10
    if x is None:
        y += 1
    else:
        y -= 1
    return y
a = paddle.to_tensor(1)
b = paddle.to_tensor(2)
c = paddle.to_tensor(3)
d = paddle.to_tensor(4)
true_tensor = paddle.to_tensor(True)
false_tensor = paddle.to_tensor(False)

class TestExecutor(TestCaseBase):

    def test_simple(self):
        if False:
            print('Hello World!')
        self.assert_results(jump_absolute, 5, a)
        self.assert_results(pop_jump_if_false, True, a)
        self.assert_results(pop_jump_if_false, False, a)
        self.assert_results(jump_if_false_or_pop, True, a)
        self.assert_results(jump_if_false_or_pop, False, a)
        self.assert_results(jump_if_true_or_pop, True, a)
        self.assert_results(jump_if_true_or_pop, False, a)
        self.assert_results(pop_jump_if_true, True, False, a)
        self.assert_results(pop_jump_if_true, False, False, a)
        self.assert_results(pop_jump_if_none, None, a)
        self.assert_results(pop_jump_if_none, True, a)
        self.assert_results(pop_jump_if_not_none, None, a)
        self.assert_results(pop_jump_if_not_none, True, a)

    def test_breakgraph(self):
        if False:
            print('Hello World!')
        self.assert_results(pop_jump_if_false, true_tensor, a)
        self.assert_results(jump_if_false_or_pop, true_tensor, a)
        self.assert_results(jump_if_true_or_pop, false_tensor, a)
        self.assert_results(pop_jump_if_true, true_tensor, false_tensor, a)
        self.assert_results(jump_absolute, 5, a)
        self.assert_results(pop_jump_if_false, false_tensor, a)
        self.assert_results(jump_if_false_or_pop, false_tensor, a)
        self.assert_results(jump_if_true_or_pop, false_tensor, a)
        self.assert_results(pop_jump_if_true, true_tensor, false_tensor, a)
        self.assert_results(pop_jump_if_none, true_tensor, a)
        self.assert_results(pop_jump_if_not_none, true_tensor, a)
if __name__ == '__main__':
    unittest.main()