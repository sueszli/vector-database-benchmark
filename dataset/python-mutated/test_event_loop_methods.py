import _testcapi
import asyncio
import unittest

class EventLoopMethodsTestCase(unittest.TestCase):

    def test_call_soon_calls(self):
        if False:
            return 10
        get_debug_called = False
        args_info = []
        capture_arg = False

        class Loop:

            def get_debug_impl(self):
                if False:
                    i = 10
                    return i + 15
                nonlocal get_debug_called
                get_debug_called = True
                return False

            def call_soon_impl(self, args, kwargs):
                if False:
                    while True:
                        i = 10
                if capture_arg:
                    self.captured = (args, kwargs)
                args_info.append((id(args), id(kwargs)))
        Loop.get_debug = _testcapi.make_get_debug_descriptor(Loop)
        Loop.call_soon = _testcapi.make_call_soon_descriptor(Loop)
        loop = Loop()
        loop.__class__
        fut = asyncio.Future(loop=loop)
        self.assertTrue(get_debug_called)
        fut.set_result(10)
        fut.add_done_callback(lambda *args: 0)
        fut.add_done_callback(lambda *args: 0)
        self.assertEqual(args_info[0], args_info[1])
        capture_arg = True
        args_info = []
        fut.add_done_callback(lambda *args: 0)
        fut.add_done_callback(lambda *args: 0)
        self.assertNotEqual(args_info[0][0], args_info[1][0])
        self.assertNotEqual(args_info[0][1], args_info[1][1])