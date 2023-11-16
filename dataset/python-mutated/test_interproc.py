import gc
from numba import jit, int32
import unittest

def foo(a, b):
    if False:
        print('Hello World!')
    return a + b

def bar(a, b):
    if False:
        print('Hello World!')
    return cfoo(a, b) + b

@jit
def inner(x, y):
    if False:
        return 10
    return x + y

@jit(nopython=True)
def outer(x, y):
    if False:
        for i in range(10):
            print('nop')
    return inner(x, y)

class TestInterProc(unittest.TestCase):

    def test_bar_call_foo(self):
        if False:
            while True:
                i = 10
        global cfoo
        cfoo = jit((int32, int32), nopython=True)(foo)
        cbar = jit((int32, int32), nopython=True)(bar)
        self.assertEqual(cbar(1, 2), 1 + 2 + 2)

    def test_bar_call_foo_compiled_twice(self):
        if False:
            return 10
        global cfoo
        for i in range(2):
            cfoo = jit((int32, int32), nopython=True)(foo)
            gc.collect()
        cbar = jit((int32, int32), nopython=True)(bar)
        self.assertEqual(cbar(1, 2), 1 + 2 + 2)

    def test_callsite_compilation(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(outer(1, 2), 1 + 2)
if __name__ == '__main__':
    unittest.main()