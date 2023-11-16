class Base:

    def __init__(self):
        if False:
            return 10
        self.a = 1

    def meth(self):
        if False:
            for i in range(10):
                print('nop')
        print('in Base meth', self.a)

class Sub(Base):

    def meth(self):
        if False:
            print('Hello World!')
        print('in Sub meth')
        return super().meth()
a = Sub()
a.meth()

class A:

    def p(self):
        if False:
            while True:
                i = 10
        print(str(super())[:18])
A().p()

class A:
    bar = 123

    def foo(self):
        if False:
            while True:
                i = 10
        print('A foo')
        return [1, 2, 3]

class B(A):

    def foo(self):
        if False:
            return 10
        print('B foo')
        print(super().bar)
        return super().foo().count(2)
print(B().foo())
try:
    super(1, 1)
except TypeError:
    print('TypeError')
assert hasattr(super(B, B()), 'foo')
try:
    super(B, B()).foo = 1
except AttributeError:
    print('AttributeError')
try:
    del super(B, B()).foo
except AttributeError:
    print('AttributeError')