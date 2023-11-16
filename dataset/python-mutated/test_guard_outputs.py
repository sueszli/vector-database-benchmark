from __future__ import annotations
import unittest
from test_case_base import TestCaseBase, test_instruction_translator_cache_context
import paddle

def non_operator_related_fn(x: int, y: int):
    if False:
        while True:
            i = 10
    return x + y

def partial_non_operator_related_fn(x: paddle.Tensor, y: paddle.Tensor, z: int):
    if False:
        i = 10
        return i + 15
    a = x + y
    return [a, z + z]

def guard_inputs(x: int, y: int, z: int):
    if False:
        i = 10
        return i + 15
    return x + y + z

class TestGuardOutputs(TestCaseBase):

    def test_non_operator_related_fn(self):
        if False:
            while True:
                i = 10
        with test_instruction_translator_cache_context() as ctx:
            self.assert_results(non_operator_related_fn, 1, 2)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(non_operator_related_fn, 3, 4)
            self.assertEqual(ctx.translate_count, 2)

    def test_partial_non_operator_related_fn(self):
        if False:
            i = 10
            return i + 15
        with test_instruction_translator_cache_context() as ctx:
            self.assert_results(partial_non_operator_related_fn, paddle.to_tensor(1), paddle.to_tensor(2), 3)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(partial_non_operator_related_fn, paddle.to_tensor(4), paddle.to_tensor(5), 6)
            self.assertEqual(ctx.translate_count, 2)

    def test_guard_inputs(self):
        if False:
            print('Hello World!')
        with test_instruction_translator_cache_context() as ctx:
            self.assert_results(guard_inputs, 1, 2, 3)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(guard_inputs, 0, 2, 3)
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(guard_inputs, 1, 0, 3)
            self.assertEqual(ctx.translate_count, 3)
            self.assert_results(guard_inputs, 1, 2, 0)
            self.assertEqual(ctx.translate_count, 4)
if __name__ == '__main__':
    unittest.main()