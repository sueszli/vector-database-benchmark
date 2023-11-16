from Cython.TestUtils import CythonTest

class TestCodeWriter(CythonTest):

    def t(self, codestr):
        if False:
            print('Hello World!')
        self.assertCode(codestr, self.fragment(codestr).root)

    def test_print(self):
        if False:
            return 10
        self.t(u'\n                    print(x + y ** 2)\n                    print(x, y, z)\n                    print(x + y, x + y * z, x * (y + z))\n               ')

    def test_if(self):
        if False:
            for i in range(10):
                print('nop')
        self.t(u'if x:\n    pass')

    def test_ifelifelse(self):
        if False:
            return 10
        self.t(u'\n                    if x:\n                        pass\n                    elif y:\n                        pass\n                    elif z + 34 ** 34 - 2:\n                        pass\n                    else:\n                        pass\n                ')

    def test_def(self):
        if False:
            print('Hello World!')
        self.t(u'\n                    def f(x, y, z):\n                        pass\n                    def f(x = 34, y = 54, z):\n                        pass\n               ')

    def test_cdef(self):
        if False:
            for i in range(10):
                print('nop')
        self.t(u'\n                    cdef f(x, y, z):\n                        pass\n                    cdef public void (x = 34, y = 54, z):\n                        pass\n                    cdef f(int *x, void *y, Value *z):\n                        pass\n                    cdef f(int **x, void **y, Value **z):\n                        pass\n                    cdef inline f(int &x, Value &z):\n                        pass\n               ')

    def test_longness_and_signedness(self):
        if False:
            print('Hello World!')
        self.t(u'def f(unsigned long long long long long int y):\n    pass')

    def test_signed_short(self):
        if False:
            for i in range(10):
                print('nop')
        self.t(u'def f(signed short int y):\n    pass')

    def test_typed_args(self):
        if False:
            while True:
                i = 10
        self.t(u'def f(int x, unsigned long int y):\n    pass')

    def test_cdef_var(self):
        if False:
            return 10
        self.t(u'\n                    cdef int hello\n                    cdef int hello = 4, x = 3, y, z\n                ')

    def test_for_loop(self):
        if False:
            i = 10
            return i + 15
        self.t(u'\n                    for x, y, z in f(g(h(34) * 2) + 23):\n                        print(x, y, z)\n                    else:\n                        print(43)\n                ')
        self.t(u'\n                    for abc in (1, 2, 3):\n                        print(x, y, z)\n                    else:\n                        print(43)\n                ')

    def test_while_loop(self):
        if False:
            return 10
        self.t(u'\n                    while True:\n                        while True:\n                            while True:\n                                continue\n                ')

    def test_inplace_assignment(self):
        if False:
            i = 10
            return i + 15
        self.t(u'x += 43')

    def test_cascaded_assignment(self):
        if False:
            for i in range(10):
                print('nop')
        self.t(u'x = y = z = abc = 43')

    def test_attribute(self):
        if False:
            i = 10
            return i + 15
        self.t(u'a.x')

    def test_return_none(self):
        if False:
            for i in range(10):
                print('nop')
        self.t(u'\n                    def f(x, y, z):\n                        return\n                    cdef f(x, y, z):\n                        return\n                    def f(x, y, z):\n                        return None\n                    cdef f(x, y, z):\n                        return None\n                    def f(x, y, z):\n                        return 1234\n                    cdef f(x, y, z):\n                        return 1234\n               ')
if __name__ == '__main__':
    import unittest
    unittest.main()