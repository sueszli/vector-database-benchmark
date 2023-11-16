"""Unit tests for zero-argument super() & related machinery."""
import unittest

class A:

    def f(self):
        if False:
            print('Hello World!')
        return 'A'

    @classmethod
    def cm(cls):
        if False:
            print('Hello World!')
        return (cls, 'A')

class B(A):

    def f(self):
        if False:
            i = 10
            return i + 15
        return super().f() + 'B'

    @classmethod
    def cm(cls):
        if False:
            while True:
                i = 10
        return (cls, super().cm(), 'B')

class C(A):

    def f(self):
        if False:
            return 10
        return super().f() + 'C'

    @classmethod
    def cm(cls):
        if False:
            for i in range(10):
                print('nop')
        return (cls, super().cm(), 'C')

class D(C, B):

    def f(self):
        if False:
            return 10
        return super().f() + 'D'

    def cm(cls):
        if False:
            while True:
                i = 10
        return (cls, super().cm(), 'D')

class E(D):
    pass

class F(E):
    f = E.f

class G(A):
    pass

class TestSuper(unittest.TestCase):

    def tearDown(self):
        if False:
            print('Hello World!')
        nonlocal __class__
        __class__ = TestSuper

    def test_basics_working(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(D().f(), 'ABCD')

    def test_class_getattr_working(self):
        if False:
            print('Hello World!')
        self.assertEqual(D.f(D()), 'ABCD')

    def test_subclass_no_override_working(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(E().f(), 'ABCD')
        self.assertEqual(E.f(E()), 'ABCD')

    def test_unbound_method_transfer_working(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(F().f(), 'ABCD')
        self.assertEqual(F.f(F()), 'ABCD')

    def test_class_methods_still_working(self):
        if False:
            print('Hello World!')
        self.assertEqual(A.cm(), (A, 'A'))
        self.assertEqual(A().cm(), (A, 'A'))
        self.assertEqual(G.cm(), (G, 'A'))
        self.assertEqual(G().cm(), (G, 'A'))

    def test_super_in_class_methods_working(self):
        if False:
            i = 10
            return i + 15
        d = D()
        self.assertEqual(d.cm(), (d, (D, (D, (D, 'A'), 'B'), 'C'), 'D'))
        e = E()
        self.assertEqual(e.cm(), (e, (E, (E, (E, 'A'), 'B'), 'C'), 'D'))

    def test_super_with_closure(self):
        if False:
            print('Hello World!')

        class E(A):

            def f(self):
                if False:
                    i = 10
                    return i + 15

                def nested():
                    if False:
                        while True:
                            i = 10
                    self
                return super().f() + 'E'
        self.assertEqual(E().f(), 'AE')

    def test_various___class___pathologies(self):
        if False:
            for i in range(10):
                print('nop')

        class X(A):

            def f(self):
                if False:
                    for i in range(10):
                        print('nop')
                return super().f()
            __class__ = 413
        x = X()
        self.assertEqual(x.f(), 'A')
        self.assertEqual(x.__class__, 413)

        class X:
            x = __class__

            def f():
                if False:
                    i = 10
                    return i + 15
                __class__
        self.assertIs(X.x, type(self))
        with self.assertRaises(NameError) as e:
            exec('class X:\n                __class__\n                def f():\n                    __class__', globals(), {})
        self.assertIs(type(e.exception), NameError)

        class X:
            global __class__
            __class__ = 42

            def f():
                if False:
                    return 10
                __class__
        self.assertEqual(globals()['__class__'], 42)
        del globals()['__class__']
        self.assertNotIn('__class__', X.__dict__)

        class X:
            nonlocal __class__
            __class__ = 42

            def f():
                if False:
                    for i in range(10):
                        print('nop')
                __class__
        self.assertEqual(__class__, 42)

    def test___class___instancemethod(self):
        if False:
            for i in range(10):
                print('nop')

        class X:

            def f(self):
                if False:
                    while True:
                        i = 10
                return __class__
        self.assertIs(X().f(), X)

    def test___class___classmethod(self):
        if False:
            while True:
                i = 10

        class X:

            @classmethod
            def f(cls):
                if False:
                    while True:
                        i = 10
                return __class__
        self.assertIs(X.f(), X)

    def test___class___staticmethod(self):
        if False:
            while True:
                i = 10

        class X:

            @staticmethod
            def f():
                if False:
                    print('Hello World!')
                return __class__
        self.assertIs(X.f(), X)

    def test___class___new(self):
        if False:
            for i in range(10):
                print('nop')
        test_class = None

        class Meta(type):

            def __new__(cls, name, bases, namespace):
                if False:
                    i = 10
                    return i + 15
                nonlocal test_class
                self = super().__new__(cls, name, bases, namespace)
                test_class = self.f()
                return self

        class A(metaclass=Meta):

            @staticmethod
            def f():
                if False:
                    for i in range(10):
                        print('nop')
                return __class__
        self.assertIs(test_class, A)

    def test___class___delayed(self):
        if False:
            while True:
                i = 10
        test_namespace = None

        class Meta(type):

            def __new__(cls, name, bases, namespace):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal test_namespace
                test_namespace = namespace
                return None

        class A(metaclass=Meta):

            @staticmethod
            def f():
                if False:
                    for i in range(10):
                        print('nop')
                return __class__
        self.assertIs(A, None)
        B = type('B', (), test_namespace)
        self.assertIs(B.f(), B)

    def test___class___mro(self):
        if False:
            return 10
        test_class = None

        class Meta(type):

            def mro(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.__dict__['f']()
                return super().mro()

        class A(metaclass=Meta):

            def f():
                if False:
                    print('Hello World!')
                nonlocal test_class
                test_class = __class__
        self.assertIs(test_class, A)

    def test___classcell___expected_behaviour(self):
        if False:
            return 10

        class Meta(type):

            def __new__(cls, name, bases, namespace):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal namespace_snapshot
                namespace_snapshot = namespace.copy()
                return super().__new__(cls, name, bases, namespace)
        namespace_snapshot = None

        class WithoutClassRef(metaclass=Meta):
            pass
        self.assertNotIn('__classcell__', namespace_snapshot)
        namespace_snapshot = None

        class WithClassRef(metaclass=Meta):

            def f(self):
                if False:
                    while True:
                        i = 10
                return __class__
        class_cell = namespace_snapshot['__classcell__']
        method_closure = WithClassRef.f.__closure__
        self.assertEqual(len(method_closure), 1)
        self.assertIs(class_cell, method_closure[0])
        with self.assertRaises(AttributeError):
            WithClassRef.__classcell__

    def test___classcell___missing(self):
        if False:
            return 10

        class Meta(type):

            def __new__(cls, name, bases, namespace):
                if False:
                    i = 10
                    return i + 15
                namespace.pop('__classcell__', None)
                return super().__new__(cls, name, bases, namespace)

        class WithoutClassRef(metaclass=Meta):
            pass
        expected_error = '__class__ not set.*__classcell__ propagated'
        with self.assertRaisesRegex(RuntimeError, expected_error):

            class WithClassRef(metaclass=Meta):

                def f(self):
                    if False:
                        return 10
                    return __class__

    def test___classcell___overwrite(self):
        if False:
            print('Hello World!')

        class Meta(type):

            def __new__(cls, name, bases, namespace, cell):
                if False:
                    return 10
                namespace['__classcell__'] = cell
                return super().__new__(cls, name, bases, namespace)
        for bad_cell in (None, 0, '', object()):
            with self.subTest(bad_cell=bad_cell):
                with self.assertRaises(TypeError):

                    class A(metaclass=Meta, cell=bad_cell):
                        pass

    def test___classcell___wrong_cell(self):
        if False:
            print('Hello World!')

        class Meta(type):

            def __new__(cls, name, bases, namespace):
                if False:
                    print('Hello World!')
                cls = super().__new__(cls, name, bases, namespace)
                B = type('B', (), namespace)
                return cls
        with self.assertRaises(TypeError):

            class A(metaclass=Meta):

                def f(self):
                    if False:
                        while True:
                            i = 10
                    return __class__

    def test_obscure_super_errors(self):
        if False:
            i = 10
            return i + 15

        def f():
            if False:
                while True:
                    i = 10
            super()
        self.assertRaises(RuntimeError, f)

        def f(x):
            if False:
                while True:
                    i = 10
            del x
            super()
        self.assertRaises(RuntimeError, f, None)

        class X:

            def f(x):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal __class__
                del __class__
                super()
        self.assertRaises(RuntimeError, X().f)

    def test_cell_as_self(self):
        if False:
            while True:
                i = 10

        class X:

            def meth(self):
                if False:
                    return 10
                super()

        def f():
            if False:
                while True:
                    i = 10
            k = X()

            def g():
                if False:
                    return 10
                return k
            return g
        c = f().__closure__[0]
        self.assertRaises(TypeError, X.meth, c)

    def test_super_init_leaks(self):
        if False:
            print('Hello World!')
        sp = super(float, 1.0)
        for i in range(1000):
            super.__init__(sp, int, i)

    def test_unusual_getattro(self):
        if False:
            i = 10
            return i + 15

        class MyType(type):
            pass
        mytype = MyType('foo', (MyType,), {})
        super(MyType, type(mytype)).__setattr__(mytype, 'bar', 1)
        self.assertEqual(mytype.bar, 1)
if __name__ == '__main__':
    unittest.main()