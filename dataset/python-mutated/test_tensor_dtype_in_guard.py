import sys
import unittest
from test_case_base import TestCaseBase, test_instruction_translator_cache_context
import paddle
from paddle.jit import sot
from paddle.jit.sot.utils import strict_mode_guard

def foo(x, y):
    if False:
        return 10
    if x.dtype == paddle.float32:
        out = x + y
    else:
        out = x - y
    return out

def dtype_in_guard(x, y):
    if False:
        return 10
    sot.psdb.fallback()
    with paddle.amp.auto_cast(level='O2'):
        for i in range(10):
            z = foo(x, y)
            x = z
        return x

def bar(x, y):
    if False:
        return 10
    if x == paddle.float32:
        return y + 1
    else:
        return y - 1

def dtype_as_input(x, y):
    if False:
        for i in range(10):
            print('nop')
    sot.psdb.fallback()
    with paddle.amp.auto_cast(level='O2'):
        for i in range(10):
            z = bar(x, y)
            y = z
        return y

class TestDtypeInGuard(TestCaseBase):

    @strict_mode_guard(False)
    def test_dtype_in_guard(self):
        if False:
            print('Hello World!')
        with test_instruction_translator_cache_context() as ctx:
            x = paddle.to_tensor([2], dtype='float32')
            y = paddle.to_tensor([3], dtype='float32')
            self.assert_results(dtype_in_guard, x, y)
            if sys.version_info >= (3, 11):
                self.assertEqual(ctx.translate_count, 1)
            else:
                self.assertEqual(ctx.translate_count, 2)

    @strict_mode_guard(False)
    def test_input_dtype_in_guard(self):
        if False:
            print('Hello World!')
        with test_instruction_translator_cache_context() as ctx:
            x = paddle.float32
            y = paddle.to_tensor([3], dtype='float32')
            self.assert_results(dtype_as_input, x, y)
            if sys.version_info >= (3, 11):
                self.assertEqual(ctx.translate_count, 1)
            else:
                self.assertEqual(ctx.translate_count, 2)
if __name__ == '__main__':
    unittest.main()