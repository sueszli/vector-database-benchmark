from builtins import _test_sink, _test_source

def tito_zero(x):
    if False:
        i = 10
        return i + 15
    return x

def tito_one(x):
    if False:
        print('Hello World!')
    return tito_zero(x)

def tito_two(x):
    if False:
        return 10
    return tito_one(x)

def tito_three(x):
    if False:
        return 10
    return tito_two(x)

def tito_max_consecutive(x):
    if False:
        print('Hello World!')
    a = tito_zero(x)
    b = tito_two(a)
    c = tito_one(b)
    return c

def tito_min_disjoint(x, y):
    if False:
        return 10
    if x:
        return tito_zero(x)
    else:
        return tito_one(x)

def tito_min_disjoint_max_consecutive(x, y):
    if False:
        while True:
            i = 10
    if y:
        a = tito_one(x)
        b = tito_zero(a)
    else:
        a = tito_two(x)
        b = tito_zero(a)
    return b

class C:

    def tito(self, parameter):
        if False:
            while True:
                i = 10
        ...

def tito_obscure(x):
    if False:
        print('Hello World!')
    c = C()
    return c.tito(x)

def tito_four(x):
    if False:
        return 10
    return tito_three(x)

def issue():
    if False:
        while True:
            i = 10
    x = _test_source()
    y = tito_three(x)
    _test_sink(y)

def non_issue():
    if False:
        i = 10
        return i + 15
    x = _test_source()
    y = tito_four(x)
    _test_sink(y)