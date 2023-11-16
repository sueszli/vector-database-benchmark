import unittest
import sys

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
            i = 10
            return i + 15
        a = getBCppObj()
        self.assertEqual(type(a), A)
        c = getCCppObj()
        self.assertEqual(type(c), C)

    def test_factory(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(type(factory(0)), A)
        self.assertEqual(type(factory(1)), A)
        self.assertEqual(type(factory(2)), C)

    def test_return_py(self):
        if False:
            i = 10
            return i + 15

        class X(A):

            def f(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 'X.f'
        x = X()
        self.assertEqual('X.f', x.f())
        self.assertEqual('X.f', call_f(x))

    def test_self_default(self):
        if False:
            print('Hello World!')

        class X(A):

            def f(self):
                if False:
                    print('Hello World!')
                return 'X.f() -> ' + A.f(self)
        x = X()
        self.assertEqual('X.f() -> A::f()', x.f())

    def test_wrapper_downcast(self):
        if False:
            while True:
                i = 10
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
                    print('Hello World!')
                return 'R.f'
        r = R()
        self.assertEqual('R.f', r.f())

def test():
    if False:
        print('Hello World!')
    import sys
    sys.argv = [x for x in sys.argv if x != '--broken-auto-ptr']
    unittest.main()
if __name__ == '__main__':
    from polymorphism2_ext import *
    test()
else:
    from polymorphism2_auto_ptr_ext import *