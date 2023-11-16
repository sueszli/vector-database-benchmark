from builtins import _test_sink, _test_source

class Base:
    pass

class A(Base):

    def foo(self, x):
        if False:
            i = 10
            return i + 15
        return 0

    def bar(self, x):
        if False:
            return 10
        return 1

class B(Base):

    def foo(self, x):
        if False:
            i = 10
            return i + 15
        return 0

    def bar(self, x):
        if False:
            return 10
        return 1

class C(Base):

    def foo(self, x):
        if False:
            for i in range(10):
                print('nop')
        return 0

    def bar(self, x):
        if False:
            while True:
                i = 10
        return 1

class DoesNotExtendsBase:

    def foo(self, x):
        if False:
            i = 10
            return i + 15
        return 0

    def bar(self, x):
        if False:
            while True:
                i = 10
        return 1

def decorated(f, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    pass

@decorated
class X:

    def foo(self, x):
        if False:
            while True:
                i = 10
        return 0

    def bar(self, x):
        if False:
            return 10
        return 1

@decorated
class Y:

    def foo(self, x):
        if False:
            return 10
        return 0

    def bar(self, x):
        if False:
            for i in range(10):
                print('nop')
        return 1

class Z:

    def foo(self, x):
        if False:
            return 10
        return 0

    def bar(self, x):
        if False:
            print('Hello World!')
        return 1

class Table:
    pass

class FooTable(Table):

    def attribute_x(self):
        if False:
            for i in range(10):
                print('nop')
        return 0

    def attribute_y(self):
        if False:
            print('Hello World!')
        return 0

class BarTable(Table):

    def attribute_x(self):
        if False:
            print('Hello World!')
        return 0

    def attribute_z(self):
        if False:
            for i in range(10):
                print('nop')
        return 0

    def non_attribute_t(self):
        if False:
            return 10
        return 0