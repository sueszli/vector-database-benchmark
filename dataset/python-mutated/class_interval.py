from abc import abstractmethod
from builtins import _test_sink, _test_source
from typing import Union
'\n  A0\n /  B0   C0\n'

class A0:

    def m1(self, x):
        if False:
            i = 10
            return i + 15
        self.m2(x)

    def m2(self, x):
        if False:
            while True:
                i = 10
        pass

class B0(A0):

    def m0(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.m1(x)

class C0(A0):

    def m2(self, x):
        if False:
            while True:
                i = 10
        _test_sink(x)

def canonical_example(b: B0):
    if False:
        i = 10
        return i + 15
    b.m0(_test_source())
'\n  A1\n /  B1   C1\n'

class A1:

    def m1(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.m2(x)

    def m2(self, x):
        if False:
            while True:
                i = 10
        pass

class B1(A1):

    def m0(self, x):
        if False:
            print('Hello World!')
        self.m1(x)

    def m1(self, x):
        if False:
            i = 10
            return i + 15
        pass

class C1(A1):

    def m2(self, x):
        if False:
            while True:
                i = 10
        _test_sink(x)

def no_call_to_parent_class(b: B1):
    if False:
        print('Hello World!')
    b.m0(_test_source())
'\n  A2\n /  B2   C2\n \\  /\n  D2\n'

class A2:

    def m1(self, x):
        if False:
            i = 10
            return i + 15
        self.m2(x)

    def m2(self, x):
        if False:
            return 10
        pass

class B2(A2):

    def m0(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.m1(x)

class C2(A2):

    def m2(self, x):
        if False:
            print('Hello World!')
        pass

class D2(B2, C2):

    def m2(self, x):
        if False:
            while True:
                i = 10
        _test_sink(x)

def multiple_inheritance(b: B2):
    if False:
        while True:
            i = 10
    b.m0(_test_source())
'\n  A3\n  |\n  B3\n /  C3  D3\n'

class A3:

    def m1(self, x):
        if False:
            i = 10
            return i + 15
        self.m2(x)

    def m2(self, x):
        if False:
            while True:
                i = 10
        pass

class B3(A3):

    def m0(self, x):
        if False:
            print('Hello World!')
        self.m1(x)

class C3(B3):

    def m0(self, x):
        if False:
            return 10
        self.m1(x)

    def m2(self, x):
        if False:
            for i in range(10):
                print('nop')
        _test_sink(x)

class D3(B3):

    def m0(self, x):
        if False:
            print('Hello World!')
        pass

    def m2(self, x):
        if False:
            print('Hello World!')
        pass

def sink_in_subclass(b: B3):
    if False:
        return 10
    b.m0(_test_source())
'\n  A4\n /  B4   C4\n|\nD4\n'

class A4:

    def m2(self, x):
        if False:
            return 10
        self.m3(x)

    def m3(self, x):
        if False:
            for i in range(10):
                print('nop')
        pass

class B4(A4):

    def m1(self, x):
        if False:
            print('Hello World!')
        self.m2(x)

class C4(A4):

    def m3(self, x):
        if False:
            return 10
        _test_sink(x)

class D4(B4):

    def m0(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.m1(x)

def source_two_hops(d: D4):
    if False:
        return 10
    d.m0(_test_source())
'\n  A5\n /  B5   C5\n     |\n     D5\n'

class A5:

    def m1(self, x):
        if False:
            print('Hello World!')
        self.m2(x)

    def m2(self, x):
        if False:
            i = 10
            return i + 15
        pass

class B5(A5):

    def m0(self, x):
        if False:
            while True:
                i = 10
        self.m1(x)

class C5(A5):
    pass

class D5(C5):

    def m2(self, x):
        if False:
            return 10
        _test_sink(x)

def sink_two_hops(b: B5):
    if False:
        i = 10
        return i + 15
    b.m0(_test_source())
'\n   A6: [1,8]\n      /       /    B6: [2,5]   C6: [6,7]\n  |\n  |\nD6: [3,4]\n\nE6: [9,10]\n'

class A6:

    def m1(self):
        if False:
            print('Hello World!')
        return self.m0()

    def m0(self):
        if False:
            while True:
                i = 10
        pass

class B6(A6):

    def m0(self):
        if False:
            i = 10
            return i + 15
        if 1 == 1:
            return _test_source()
        else:
            return E6().m3()

class C6(A6):

    def m2(self):
        if False:
            for i in range(10):
                print('nop')
        return self.m1()

class D6(B6):

    def m0(self):
        if False:
            for i in range(10):
                print('nop')
        super().m0()

class E6:

    def m3(self):
        if False:
            while True:
                i = 10
        return _test_source()

def propagate_source_empty(c: C6):
    if False:
        print('Hello World!')
    return _test_sink(c.m1())
'\nA7: [1,2]\nB7: [3,4]\n'

class B7:

    def foo(self):
        if False:
            for i in range(10):
                print('nop')
        return self.bar()

    def bar(self):
        if False:
            print('Hello World!')
        return _test_source()

class A7:

    def bar(self, x):
        if False:
            while True:
                i = 10
        return x

    def f(self, b: B7):
        if False:
            while True:
                i = 10
        y = b.foo()
        return y
'\nA8: [1,2]\nB8: [3,4]\nC8: [5,6]\n'

class B8:

    def foo(self, x):
        if False:
            return 10
        return _test_source()

class C8:

    def foo(self, x):
        if False:
            return 10
        pass

class A8:

    def bar(self, b: Union[B8, C8], x):
        if False:
            while True:
                i = 10
        if x == 1:
            return self.baz()
        elif x == 2:
            return b.foo(x)
        elif x == 3:
            return A8().baz()
        else:
            return x

    def baz(self):
        if False:
            return 10
        return _test_source()

class A9:

    def f(self):
        if False:
            while True:
                i = 10
        return _test_source()

def call_method_via_class_name(a: A9):
    if False:
        for i in range(10):
            print('nop')
    return A9.f(a)

class A10:
    f: int = 0

    def object_target(x):
        if False:
            i = 10
            return i + 15
        a = A10()
        a.f = x

class A12:

    def f(self):
        if False:
            i = 10
            return i + 15
        return self.g()

class B12(A12):

    def g(self):
        if False:
            for i in range(10):
                print('nop')
        return 0

class C12(A12):

    def g(self):
        if False:
            return 10
        return _test_source()

def undetected_issue(c: C12):
    if False:
        return 10
    x = c.f()
    _test_sink(x)

class A13:

    def f(self):
        if False:
            i = 10
            return i + 15
        return self.g()

    @abstractmethod
    def g(self):
        if False:
            return 10
        pass

class B13(A13):

    def g(self):
        if False:
            while True:
                i = 10
        return 0

class C13(A13):

    def g(self):
        if False:
            i = 10
            return i + 15
        return _test_source()

def abstract_method(b: B13, c: C13):
    if False:
        print('Hello World!')
    x = c.f()
    _test_sink(x)
    y = b.f()
    _test_sink(y)
'\n           A: [1,8]\n          /          /           /     B: [2,5]   C: [6,7] \\/ [3,4]\n        \\     /\n         \\   /\n          \\ /\n           D: [3,4]\n'

class A14:

    def m1(self):
        if False:
            while True:
                i = 10
        return self.m2()

    def m2(self):
        if False:
            while True:
                i = 10
        pass

class C14(A14):

    def m2(self):
        if False:
            print('Hello World!')
        return _test_source()

class B14(A14):

    def m0(self):
        if False:
            return 10
        return self.m1()

    def m2(self):
        if False:
            print('Hello World!')
        return 0

class D14(B14, C14):
    pass

def multi_inheritance_no_issue_one_hop(b: B14):
    if False:
        while True:
            i = 10
    _test_sink(b.m0())

class A15:

    def m1(self):
        if False:
            for i in range(10):
                print('nop')
        return self.m2()

    def m2(self):
        if False:
            print('Hello World!')
        pass

class E15(A15):

    def m2(self):
        if False:
            print('Hello World!')
        return _test_source()

class B15(A15):

    def m0(self):
        if False:
            print('Hello World!')
        return self.m1()

    def m2(self):
        if False:
            i = 10
            return i + 15
        return 0

class C15(B15):
    pass

class D15(C15, E15):
    pass

def multi_inheritance_no_issue_two_hops(b: B15):
    if False:
        print('Hello World!')
    _test_sink(b.m0())

class A16:

    def m1(self):
        if False:
            while True:
                i = 10
        return self.m2()

    def m2(self):
        if False:
            i = 10
            return i + 15
        pass

class C16(A16):

    def m2(self):
        if False:
            while True:
                i = 10
        return 0

class B16(A16):

    def m0(self):
        if False:
            i = 10
            return i + 15
        return self.m1()

    def m2(self):
        if False:
            return 10
        return 0

class D16(B16, C16):

    def m2(self):
        if False:
            i = 10
            return i + 15
        return _test_source()

def multi_inheritance_issue(b: B16):
    if False:
        i = 10
        return i + 15
    _test_sink(b.m0())