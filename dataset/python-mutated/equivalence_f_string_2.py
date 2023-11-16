def foo1():
    if False:
        return 10
    select = 'select * '
    query = f'{select} from foo'

def foo2():
    if False:
        for i in range(10):
            print('nop')
    select = 'select * '
    name = 'foo'
    query = f'{select} from foo ' + f'where name={name}'

def foo2a():
    if False:
        return 10
    select = 'select * '
    name = 'foo'
    query = f'{select} from foo where name={name}'

def foo5():
    if False:
        i = 10
        return i + 15
    num = 1
    query = f'{num} is 1'

def foo6():
    if False:
        return 10
    complex_func = foo()
    query = f'{complex_func} is foo'