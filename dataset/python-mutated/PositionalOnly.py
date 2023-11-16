def one_argument(arg, /):
    if False:
        i = 10
        return i + 15
    return arg.upper()

def three_arguments(a, b, c, /):
    if False:
        for i in range(10):
            print('nop')
    return '-'.join([a, b, c])

def with_normal(posonly, /, normal):
    if False:
        i = 10
        return i + 15
    return posonly + '-' + normal

def defaults(required, optional='default', /):
    if False:
        while True:
            i = 10
    return required + '-' + optional

def types(first: int, second: float, /):
    if False:
        for i in range(10):
            print('nop')
    return first + second

def kwargs(x, /, **y):
    if False:
        while True:
            i = 10
    return '%s, %s' % (x, ', '.join(('%s: %s' % item for item in y.items())))