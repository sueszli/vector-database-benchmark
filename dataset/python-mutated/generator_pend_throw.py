def gen():
    if False:
        i = 10
        return i + 15
    i = 0
    while 1:
        yield i
        i += 1
g = gen()
try:
    g.pend_throw
except AttributeError:
    print('SKIP')
    raise SystemExit
print(next(g))
print(next(g))
g.pend_throw(ValueError())
v = None
try:
    v = next(g)
except Exception as e:
    print('raised', repr(e))
print('ret was:', v)
g = gen()
g.pend_throw(OSError())
try:
    next(g)
except Exception as e:
    print('raised', repr(e))

def gen_next():
    if False:
        i = 10
        return i + 15
    next(g)
    yield 1
g = gen_next()
try:
    next(g)
except Exception as e:
    print('raised', repr(e))

def gen_pend_throw():
    if False:
        while True:
            i = 10
    g.pend_throw(ValueError())
    yield 1
g = gen_pend_throw()
try:
    next(g)
except Exception as e:
    print('raised', repr(e))

class CancelledError(Exception):
    pass

def gen_cancelled():
    if False:
        i = 10
        return i + 15
    for i in range(5):
        try:
            yield i
        except CancelledError:
            print('ignore CancelledError')
g = gen_cancelled()
print(next(g))
g.pend_throw(CancelledError())
print(next(g))
g = gen_cancelled()
g.pend_throw(CancelledError())
try:
    next(g)
except Exception as e:
    print('raised', repr(e))
g = gen()
next(g)
print(repr(g.pend_throw(CancelledError())))
print(repr(g.pend_throw(OSError)))
g = gen()
next(g)
g.pend_throw(CancelledError())
print(repr(g.pend_throw(None)))
print(next(g))