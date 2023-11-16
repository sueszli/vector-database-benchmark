def test_multiple_inheritance_cpp():
    if False:
        while True:
            i = 10
    from pybind11_tests import MIType
    mt = MIType(3, 4)
    assert mt.foo() == 3
    assert mt.bar() == 4

def test_multiple_inheritance_mix1():
    if False:
        for i in range(10):
            print('nop')
    from pybind11_tests import Base2

    class Base1:

        def __init__(self, i):
            if False:
                print('Hello World!')
            self.i = i

        def foo(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.i

    class MITypePy(Base1, Base2):

        def __init__(self, i, j):
            if False:
                return 10
            Base1.__init__(self, i)
            Base2.__init__(self, j)
    mt = MITypePy(3, 4)
    assert mt.foo() == 3
    assert mt.bar() == 4

def test_multiple_inheritance_mix2():
    if False:
        i = 10
        return i + 15
    from pybind11_tests import Base1

    class Base2:

        def __init__(self, i):
            if False:
                return 10
            self.i = i

        def bar(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.i

    class MITypePy(Base1, Base2):

        def __init__(self, i, j):
            if False:
                return 10
            Base1.__init__(self, i)
            Base2.__init__(self, j)
    mt = MITypePy(3, 4)
    assert mt.foo() == 3
    assert mt.bar() == 4

def test_multiple_inheritance_virtbase():
    if False:
        while True:
            i = 10
    from pybind11_tests import Base12a, bar_base2a, bar_base2a_sharedptr

    class MITypePy(Base12a):

        def __init__(self, i, j):
            if False:
                i = 10
                return i + 15
            Base12a.__init__(self, i, j)
    mt = MITypePy(3, 4)
    assert mt.bar() == 4
    assert bar_base2a(mt) == 4
    assert bar_base2a_sharedptr(mt) == 4