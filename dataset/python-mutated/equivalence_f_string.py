def foo1():
    if False:
        for i in range(10):
            print('nop')
    w = 'foo'
    query = f'hello {w}'

def foo2():
    if False:
        print('Hello World!')
    ww = 'bar'
    query = f'ASD{ww}'

def foo3():
    if False:
        i = 10
        return i + 15
    www = 'bar'
    query = f'SELECT {www}'

def foo4():
    if False:
        i = 10
        return i + 15
    ww = 'foo'
    www = 'bar'
    query = f'SELECT {www} and {ww}'

def foo5():
    if False:
        i = 10
        return i + 15
    num = 1
    query = f'num = {num}'

def foo6():
    if False:
        for i in range(10):
            print('nop')
    complex_func = foo()
    query = f'complex = {complex_func}'