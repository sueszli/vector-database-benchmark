import numpy as np
import numba
from numba.tests.support import TestCase

class Issue455(object):
    """
    Test code from issue 455.
    """

    def __init__(self):
        if False:
            return 10
        self.f = []

    def create_f(self):
        if False:
            print('Hello World!')
        code = '\n        def f(x):\n            n = x.shape[0]\n            for i in range(n):\n                x[i] = 1.\n        '
        d = {}
        exec(code.strip(), d)
        self.f.append(numba.jit('void(f8[:])', nopython=True)(d['f']))

    def call_f(self):
        if False:
            for i in range(10):
                print('nop')
        a = np.zeros(10)
        for f in self.f:
            f(a)
        return a

class TestDynFunc(TestCase):

    def test_issue_455(self):
        if False:
            print('Hello World!')
        inst = Issue455()
        inst.create_f()
        a = inst.call_f()
        self.assertPreciseEqual(a, np.ones_like(a))
if __name__ == '__main__':
    unittest.main()