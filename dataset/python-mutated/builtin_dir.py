print('__name__' in dir())
import sys
print('version' in dir(sys))
print('append' in dir(list))

class Foo:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.x = 1
foo = Foo()
print('__init__' in dir(foo))
print('x' in dir(foo))

class A:

    def a():
        if False:
            i = 10
            return i + 15
        pass

class B(A):

    def b():
        if False:
            while True:
                i = 10
        pass
d = dir(B())
print(d.count('a'), d.count('b'))

class C(A):

    def c():
        if False:
            for i in range(10):
                print('nop')
        pass

class D(B, C):

    def d():
        if False:
            i = 10
            return i + 15
        pass
d = dir(D())
print(d.count('a'), d.count('b'), d.count('c'), d.count('d'))