def foo1():
    if False:
        for i in range(10):
            print('nop')
    select = 'select * '
    query = f'{select} from foo'

def foo2():
    if False:
        print('Hello World!')
    select = 'select * '
    name = 'foo'
    query = f'{select} from foo where name={name}'

def foo5():
    if False:
        return 10
    num = 1
    query = f'{num} is 1'

def foo6():
    if False:
        return 10
    complex_func = foo()
    query = f'{complex_func} is foo'

def foo7():
    if False:
        while True:
            i = 10
    w = 'foo'
    query = f'hello {w}'

def foo8():
    if False:
        i = 10
        return i + 15
    ww = 'bar'
    query = f'ASD{ww}'

def foo9():
    if False:
        while True:
            i = 10
    www = 'bar'
    query = f'SELECT {www}'

def foo10():
    if False:
        print('Hello World!')
    ww = 'foo'
    ww = 'foo'
    www = 'bar'
    query = f'SELECT {www} and {ww}'

def foo11():
    if False:
        i = 10
        return i + 15
    num = 1
    query = f'num = {num}'

def foo12():
    if False:
        i = 10
        return i + 15
    complex_func = foo()
    query = f'complex = {complex_func}'