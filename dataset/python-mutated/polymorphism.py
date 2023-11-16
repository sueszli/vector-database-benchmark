import unittest
from polymorphism_ext import *

class PolymorphTest(unittest.TestCase):

    def testReturnCpp(self):
        if False:
            for i in range(10):
                print('nop')
        a = getBCppObj()
        self.assertEqual('B::f()', a.f())
        self.assertEqual('B::f()', call_f(a))
        self.assertEqual('A::f()', call_f(A()))

    def test_references(self):
        if False:
            for i in range(10):
                print('nop')
        a = getBCppObj()
        self.assertEqual(type(a), A)
        c = getCCppObj()
        self.assertEqual(type(c), C)

    def test_factory(self):
        if False:
            return 10
        self.assertEqual(type(factory(0)), A)
        self.assertEqual(type(factory(1)), A)
        self.assertEqual(type(factory(2)), C)

    def test_return_py(self):
        if False:
            print('Hello World!')

        class X(A):

            def f(self):
                if False:
                    while True:
                        i = 10
                return 'X.f'
        x = X()
        self.assertEqual('X.f', x.f())
        self.assertEqual('X.f', call_f(x))

    def test_wrapper_downcast(self):
        if False:
            print('Hello World!')
        a = pass_a(D())
        self.assertEqual('D::g()', a.g())

    def test_pure_virtual(self):
        if False:
            for i in range(10):
                print('nop')
        p = P()
        self.assertRaises(RuntimeError, p.f)
        q = Q()
        self.assertEqual('Q::f()', q.f())

        class R(P):

            def f(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 'R.f'
        r = R()
        self.assertEqual('R.f', r.f())
if __name__ == '__main__':
    import sys
    sys.argv = [x for x in sys.argv if x != '--broken-auto-ptr']
    unittest.main()