class Base:

    @classmethod
    def foo(cls):
        if False:
            i = 10
            return i + 15
        print(cls.__name__)
try:
    Base.__name__
except AttributeError:
    print('SKIP')
    raise SystemExit

class Sub(Base):
    pass
Sub.foo()

class A(object):
    foo = 0

    @classmethod
    def bar(cls):
        if False:
            while True:
                i = 10
        print(cls.foo)

    def baz(self):
        if False:
            for i in range(10):
                print('nop')
        print(self.foo)

class B(A):
    foo = 1
B.bar()
B().bar()
B().baz()