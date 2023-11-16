from __future__ import annotations
import inspect
import random
import types
import unittest
from unittest.mock import patch
from test_case_base import TestCaseBase, test_instruction_translator_cache_context
from paddle.jit.sot.opcode_translator.custom_code import CustomCode
from paddle.jit.sot.opcode_translator.executor.executor_cache import OpcodeExecutorCache

def fake_frames() -> tuple[types.FrameType, types.FrameType, types.FrameType, types.FrameType, types.FrameType]:
    if False:
        print('Hello World!')

    def fake_inner_fn_1():
        if False:
            print('Hello World!')
        frame = inspect.currentframe()
        assert frame is not None
        return frame

    def fake_inner_fn_2():
        if False:
            for i in range(10):
                print('nop')
        frame = inspect.currentframe()
        assert frame is not None
        return frame

    def fake_inner_fn_3():
        if False:
            print('Hello World!')
        frame = inspect.currentframe()
        assert frame is not None
        return frame

    def fake_inner_fn_4():
        if False:
            for i in range(10):
                print('nop')
        frame = inspect.currentframe()
        assert frame is not None
        return frame

    def fake_inner_fn_5():
        if False:
            i = 10
            return i + 15
        frame = inspect.currentframe()
        assert frame is not None
        return frame
    return (fake_inner_fn_1(), fake_inner_fn_2(), fake_inner_fn_3(), fake_inner_fn_4(), fake_inner_fn_5())
(FRAME_1, FRAME_2, FRAME_3, FRAME_4, FRAME_5) = fake_frames()

def mock_start_translate(frame: types.FrameType, **kwargs):
    if False:
        return 10
    translate_map = {FRAME_1: (CustomCode(FRAME_2.f_code, False), lambda frame: True), FRAME_3: (CustomCode(FRAME_4.f_code, False), lambda frame: False), FRAME_5: (CustomCode(None, False), lambda frame: True)}
    return translate_map[frame]

class TestOpcodeExecutorCache(unittest.TestCase):

    def reset(self):
        if False:
            print('Hello World!')
        global translate_count
        translate_count = 0
        OpcodeExecutorCache().clear()

    @patch('paddle.jit.sot.opcode_translator.executor.executor_cache.start_translate', mock_start_translate)
    def test_cache_hit(self):
        if False:
            for i in range(10):
                print('nop')
        with test_instruction_translator_cache_context() as ctx:
            translated_code_1 = OpcodeExecutorCache()(FRAME_1)
            assert translated_code_1 is not None
            self.assertEqual(translated_code_1.code, FRAME_2.f_code)
            self.assertEqual(ctx.translate_count, 1)
            translated_code_2 = OpcodeExecutorCache()(FRAME_1)
            assert translated_code_2 is not None
            self.assertEqual(translated_code_2.code, FRAME_2.f_code)
            self.assertEqual(ctx.translate_count, 1)

    @patch('paddle.jit.sot.opcode_translator.executor.executor_cache.start_translate', mock_start_translate)
    def test_cache_miss_due_to_unknown_code(self):
        if False:
            print('Hello World!')
        with test_instruction_translator_cache_context() as ctx:
            translated_code_1 = OpcodeExecutorCache()(FRAME_1)
            assert translated_code_1 is not None
            self.assertEqual(translated_code_1.code, FRAME_2.f_code)
            self.assertEqual(ctx.translate_count, 1)
            translated_code_2 = OpcodeExecutorCache()(FRAME_3)
            assert translated_code_2 is not None
            self.assertEqual(translated_code_2.code, FRAME_4.f_code)
            self.assertEqual(ctx.translate_count, 2)

    @patch('paddle.jit.sot.opcode_translator.executor.executor_cache.start_translate', mock_start_translate)
    def test_cache_miss_due_to_check_failed(self):
        if False:
            i = 10
            return i + 15
        with test_instruction_translator_cache_context() as ctx:
            translated_code_1 = OpcodeExecutorCache()(FRAME_3)
            assert translated_code_1 is not None
            self.assertEqual(translated_code_1.code, FRAME_4.f_code)
            self.assertEqual(ctx.translate_count, 1)
            translated_code_2 = OpcodeExecutorCache()(FRAME_3)
            assert translated_code_2 is not None
            self.assertEqual(translated_code_2.code, FRAME_4.f_code)
            self.assertEqual(ctx.translate_count, 2)

def foo(x):
    if False:
        print('Hello World!')
    return x + 1

class TestCacheExceedLimit(TestCaseBase):

    def test_cache_exceed_limit(self):
        if False:
            return 10
        for _ in range(30):
            input = random.random()
            self.assert_results(foo, input)
if __name__ == '__main__':
    unittest.main()