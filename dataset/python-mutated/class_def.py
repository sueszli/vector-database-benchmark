import collections

class A:
    pass

class B:
    pass

class D(A):
    pass

class Foo(A, B):
    pass

class Bar(A, B):
    pass

class Baz(collections.namedtuple('Foo', ['bar', 'baz', 'quux'])):
    pass

class Quux:
    pass

def f():
    if False:
        while True:
            i = 10
    global Quux