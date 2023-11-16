def starred_return():
    if False:
        print('Hello World!')
    my_list = ['value2', 'value3']
    return ('value1', *my_list)

def starred_yield():
    if False:
        i = 10
        return i + 15
    my_list = ['value2', 'value3']
    yield ('value1', *my_list)
a: Tuple[str, int] = ('1', 2)
a: Tuple[int, ...] = (b, *c, d)

def t():
    if False:
        print('Hello World!')
    a: str = (yield 'a')

def starred_return():
    if False:
        for i in range(10):
            print('nop')
    my_list = ['value2', 'value3']
    return ('value1', *my_list)

def starred_yield():
    if False:
        for i in range(10):
            print('nop')
    my_list = ['value2', 'value3']
    yield ('value1', *my_list)
a: Tuple[str, int] = ('1', 2)
a: Tuple[int, ...] = (b, *c, d)

def t():
    if False:
        for i in range(10):
            print('nop')
    a: str = (yield 'a')