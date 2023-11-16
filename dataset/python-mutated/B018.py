class Foo1:
    """abc"""

class Foo2:
    """abc"""
    a = 2
    'str'
    f'{int}'
    1j
    1
    1.0
    b'foo'
    True
    False
    None
    [1, 2]
    {1, 2}
    {'foo': 'bar'}

class Foo3:
    123
    a = 2
    'str'
    1

def foo1():
    if False:
        for i in range(10):
            print('nop')
    'my docstring'

def foo2():
    if False:
        for i in range(10):
            print('nop')
    'my docstring'
    a = 2
    'str'
    f'{int}'
    1j
    1
    1.0
    b'foo'
    True
    False
    None
    [1, 2]
    {1, 2}
    {'foo': 'bar'}

def foo3():
    if False:
        print('Hello World!')
    123
    a = 2
    'str'
    3

def foo4():
    if False:
        i = 10
        return i + 15
    ...

def foo5():
    if False:
        for i in range(10):
            print('nop')
    foo.bar
    object().__class__
    'foo' + 'bar'