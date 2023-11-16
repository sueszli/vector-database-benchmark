import unittest
import paddle
from paddle.jit.api import _clear_save_pre_hooks, _register_save_pre_hook, _run_save_pre_hooks
_counter = 0

class TestPreSaveHooks(unittest.TestCase):

    def test_pre_save_hook_functions(self):
        if False:
            i = 10
            return i + 15

        def fake_func(*args, **kwgs):
            if False:
                while True:
                    i = 10
            global _counter
            _counter += 1
        remove_handler = _register_save_pre_hook(fake_func)
        self.assertEqual(len(paddle.jit.api._save_pre_hooks), 1)
        self.assertTrue(paddle.jit.api._save_pre_hooks[0] is fake_func)
        remove_handler = _register_save_pre_hook(fake_func)
        self.assertEqual(len(paddle.jit.api._save_pre_hooks), 1)
        self.assertTrue(paddle.jit.api._save_pre_hooks[0] is fake_func)
        remove_handler.remove()
        self.assertEqual(len(paddle.jit.api._save_pre_hooks), 0)
        remove_handler = _register_save_pre_hook(fake_func)
        _clear_save_pre_hooks()
        self.assertEqual(len(paddle.jit.api._save_pre_hooks), 0)
        global _counter
        _counter = 0
        remove_handler = _register_save_pre_hook(fake_func)
        func_with_hook = _run_save_pre_hooks(fake_func)
        func_with_hook(None, None)
        self.assertEqual(_counter, 2)
if __name__ == '__main__':
    unittest.main()