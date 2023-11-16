import unittest
from numba.core.compiler_lock import global_compiler_lock, require_global_compiler_lock
from numba.tests.support import TestCase

class TestCompilerLock(TestCase):

    def test_gcl_as_context_manager(self):
        if False:
            print('Hello World!')
        with global_compiler_lock:
            require_global_compiler_lock()

    def test_gcl_as_decorator(self):
        if False:
            for i in range(10):
                print('nop')

        @global_compiler_lock
        def func():
            if False:
                print('Hello World!')
            require_global_compiler_lock()
        func()
if __name__ == '__main__':
    unittest.main()