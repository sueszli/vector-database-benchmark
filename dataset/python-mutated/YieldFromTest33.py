def g():
    if False:
        for i in range(10):
            print('nop')
    for a in range(3):
        yield a
    return 7

def h():
    if False:
        return 10
    yield 4
    yield 5

def f():
    if False:
        print('Hello World!')
    print('Yielded from returner', (yield g()))
    print('Yielded from non-return value', (yield h()))
print('Result', list(f()))
print('Yielder with return value', list(g()))

class Broken:

    def __iter__(self):
        if False:
            return 10
        return self

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        return 1

    def __getattr__(self, attr):
        if False:
            for i in range(10):
                print('nop')
        1 / 0

def test_broken_getattr_handling():
    if False:
        while True:
            i = 10

    def g():
        if False:
            for i in range(10):
                print('nop')
        yield from Broken()
    print('Next with send: ', end='')
    try:
        gi = g()
        next(gi)
        gi.send(1)
    except Exception as e:
        print('Caught', repr(e))
    print('Next with throw: ', end='')
    try:
        gi = g()
        next(gi)
        gi.throw(AttributeError)
    except Exception as e:
        print('Caught', repr(e))
    print('Next with close: ', end='')
    try:
        gi = g()
        next(gi)
        gi.close()
        print('All good')
    except Exception as e:
        print('Caught', repr(e))
test_broken_getattr_handling()

def test_throw_caught_subgenerator_handling():
    if False:
        for i in range(10):
            print('nop')

    def g1():
        if False:
            while True:
                i = 10
        try:
            print('Starting g1')
            yield 'g1 ham'
            yield from g2()
            yield 'g1 eggs'
        finally:
            print('Finishing g1')

    def g2():
        if False:
            i = 10
            return i + 15
        try:
            print('Starting g2')
            yield 'g2 spam'
            yield 'g2 more spam'
        except LunchError:
            print('Caught LunchError in g2')
            yield 'g2 lunch saved'
            yield 'g2 yet more spam'

    class LunchError(Exception):
        pass
    g = g1()
    for i in range(2):
        x = next(g)
        print('Yielded %s' % (x,))
    e = LunchError('tomato ejected')
    print('Throw returned', g.throw(e))
    print('Sub thrown')
    for x in g:
        print('Yielded %s' % (x,))
test_throw_caught_subgenerator_handling()

def give_cpython_generator():
    if False:
        print('Hello World!')
    return eval('(x for x in range(3))')

def gen_compiled():
    if False:
        return 10
    yield from give_cpython_generator()
    yield ...
    yield from range(7)
print('Mixing uncompiled and compiled yield from:')
print(list(gen_compiled()))