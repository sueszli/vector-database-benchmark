def f():
    if False:
        return 10
    for x in iterable:
        if check(x):
            return True
    return False

def f():
    if False:
        while True:
            i = 10
    for x in iterable:
        if check(x):
            return True
    return True

def f():
    if False:
        return 10
    for el in [1, 2, 3]:
        if is_true(el):
            return True
    raise Exception

def f():
    if False:
        while True:
            i = 10
    for x in iterable:
        if check(x):
            return False
    return True

def f():
    if False:
        for i in range(10):
            print('nop')
    for x in iterable:
        if not x.is_empty():
            return False
    return True

def f():
    if False:
        return 10
    for x in iterable:
        if check(x):
            return False
    return False

def f():
    if False:
        return 10
    for x in iterable:
        if check(x):
            return 'foo'
    return 'bar'

def f():
    if False:
        return 10
    for x in iterable:
        if check(x):
            return True
    else:
        return False

def f():
    if False:
        while True:
            i = 10
    for x in iterable:
        if check(x):
            return False
    else:
        return True

def f():
    if False:
        i = 10
        return i + 15
    for x in iterable:
        if check(x):
            return True
    else:
        return False
    return True

def f():
    if False:
        i = 10
        return i + 15
    for x in iterable:
        if check(x):
            return False
    else:
        return True
    return False

def f():
    if False:
        print('Hello World!')
    for x in iterable:
        if check(x):
            return True
        elif x.is_empty():
            return True
    return False

def f():
    if False:
        i = 10
        return i + 15
    for x in iterable:
        if check(x):
            return True
        else:
            return True
    return False

def f():
    if False:
        while True:
            i = 10
    for x in iterable:
        if check(x):
            return True
        elif x.is_empty():
            return True
    else:
        return True
    return False

def f():
    if False:
        print('Hello World!')

    def any(exp):
        if False:
            print('Hello World!')
        pass
    for x in iterable:
        if check(x):
            return True
    return False

def f():
    if False:
        return 10

    def all(exp):
        if False:
            print('Hello World!')
        pass
    for x in iterable:
        if check(x):
            return False
    return True

def f():
    if False:
        while True:
            i = 10
    x = 1
    for x in iterable:
        if check(x):
            return True
    return False

def f():
    if False:
        i = 10
        return i + 15
    x = 1
    for x in iterable:
        if check(x):
            return False
    return True

def f():
    if False:
        return 10
    for x in '012ß9💣2ℝ9012ß9💣2ℝ9012ß9💣2ℝ9012ß9💣2ℝ9012ß9💣2ℝ':
        if x.isdigit():
            return True
    return False

def f():
    if False:
        while True:
            i = 10
    for x in '012ß9💣2ℝ9012ß9💣2ℝ9012ß9💣2ℝ9012ß9💣2ℝ9012ß9💣2ℝ9':
        if x.isdigit():
            return True
    return False

async def f():
    for x in iterable:
        if await check(x):
            return True
    return False

async def f():
    for x in iterable:
        if check(x):
            return True
    return False

async def f():
    for x in await iterable:
        if check(x):
            return True
    return False

def f():
    if False:
        i = 10
        return i + 15
    for x in iterable:
        if (yield check(x)):
            return True
    return False

def f():
    if False:
        while True:
            i = 10
    for x in iterable:
        if (yield from check(x)):
            return True
    return False