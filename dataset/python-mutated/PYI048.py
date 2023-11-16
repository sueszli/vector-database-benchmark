def bar():
    if False:
        return 10
    ...

def bar():
    if False:
        while True:
            i = 10
    pass

def bar():
    if False:
        while True:
            i = 10
    'oof'

def oof():
    if False:
        i = 10
        return i + 15
    'oof'
    print('foo')

def foo():
    if False:
        i = 10
        return i + 15
    'foo'
    print('foo')
    print('foo')

def buzz():
    if False:
        print('Hello World!')
    print('fizz')
    print('buzz')
    print('test')