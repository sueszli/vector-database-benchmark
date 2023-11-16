from builtins import _test_sink, _test_source

class A:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.attribute = _test_source()

    def f1(self):
        if False:
            while True:
                i = 10
        _test_sink(self.attribute)

    def f2(self, x):
        if False:
            while True:
                i = 10
        _test_sink(x)

    def f3(self):
        if False:
            print('Hello World!')
        return _test_source()

    def f4(self):
        if False:
            print('Hello World!')
        return '1'

    def f5(self, x):
        if False:
            i = 10
            return i + 15
        pass

class B(A):

    def f1(self):
        if False:
            print('Hello World!')
        return '1'

    def f4(self):
        if False:
            return 10
        return _test_source()

    def f5(self, x):
        if False:
            print('Hello World!')
        return _test_sink(x)

    def g1(self):
        if False:
            i = 10
            return i + 15
        super(B, self).f1()
        super().f1()
        super().f2(super().f3())

    def g2(self):
        if False:
            print('Hello World!')
        super(B, self).f5(super(B, self).f4())
        super().f5(super().f4())

    def g3(self):
        if False:
            while True:
                i = 10
        self.f5(super().f4())
        super().f5(self.f4())

    def g4(self):
        if False:
            print('Hello World!')
        self.f5(self.f4())

class C(A):

    def f2(self, x):
        if False:
            i = 10
            return i + 15
        return '1'

    def g1(self):
        if False:
            for i in range(10):
                print('nop')
        return super().f1()

class D(C):

    def g1(self):
        if False:
            print('Hello World!')
        super().f1()

class E(B, A):

    def g1(self):
        if False:
            i = 10
            return i + 15
        super().f1()

    def g2(self):
        if False:
            print('Hello World!')
        super(E, self).f1()

    def g3(self):
        if False:
            return 10
        super(B, self).f1()

def attribute_B_not_overwritten():
    if False:
        print('Hello World!')
    B().g1()

def attribute_B_overwritten():
    if False:
        while True:
            i = 10
    b = B()
    b.attribute = '1'
    b.g1()

def B_overwrite_both():
    if False:
        for i in range(10):
            print('nop')
    b = B()
    b.g2()

def B_overwrite_partial():
    if False:
        print('Hello World!')
    b = B()
    b.g3()

def B_standard():
    if False:
        for i in range(10):
            print('nop')
    b = B()
    b.g4()

def attribute_C_not_overwritten():
    if False:
        while True:
            i = 10
    C().g1()

def attribute_C_overwritten():
    if False:
        while True:
            i = 10
    c = C()
    c.attribute = '1'
    c.g1()

def attribute_D_not_overwritten():
    if False:
        for i in range(10):
            print('nop')
    d = D()
    d.g1()

def attribute_D_overwritten():
    if False:
        for i in range(10):
            print('nop')
    d = D()
    d.attribute = '1'
    d.g1()

def attribute_E_not_overwritten():
    if False:
        print('Hello World!')
    e = E()
    e.g1()
    e.g2()

def attribute_E_not_overwritten_RCE():
    if False:
        i = 10
        return i + 15
    e = E()
    e.g3()

def attribute_E_overwritten():
    if False:
        while True:
            i = 10
    e = E()
    e.attribute = '1'
    e.g1()
    e.g2()
    e.g3()