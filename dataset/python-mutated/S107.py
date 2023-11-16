def ok(first, default='default'):
    if False:
        i = 10
        return i + 15
    pass

def default(first, password='default'):
    if False:
        i = 10
        return i + 15
    pass

def ok_posonly(first, /, pos, default='posonly'):
    if False:
        for i in range(10):
            print('nop')
    pass

def default_posonly(first, /, pos, password='posonly'):
    if False:
        i = 10
        return i + 15
    pass

def ok_kwonly(first, *, default='kwonly'):
    if False:
        return 10
    pass

def default_kwonly(first, *, password='kwonly'):
    if False:
        i = 10
        return i + 15
    pass

def ok_all(first, /, pos, default='posonly', *, kwonly='kwonly'):
    if False:
        return 10
    pass

def default_all(first, /, pos, secret='posonly', *, password='kwonly'):
    if False:
        while True:
            i = 10
    pass

def ok_empty(first, password=''):
    if False:
        return 10
    pass