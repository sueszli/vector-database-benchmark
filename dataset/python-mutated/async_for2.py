import sys
if sys.implementation.name == 'micropython':
    coroutine = lambda f: f
else:
    import types
    coroutine = types.coroutine

@coroutine
def f(x):
    if False:
        return 10
    print('f start:', x)
    yield (x + 1)
    yield (x + 2)
    return x + 3

class ARange:

    def __init__(self, high):
        if False:
            print('Hello World!')
        print('init')
        self.cur = 0
        self.high = high

    def __aiter__(self):
        if False:
            while True:
                i = 10
        print('aiter')
        return self

    async def __anext__(self):
        print('anext')
        print('f returned:', await f(20))
        if self.cur < self.high:
            val = self.cur
            self.cur += 1
            return val
        else:
            raise StopAsyncIteration

async def coro():
    async for x in ARange(4):
        print('x', x)
o = coro()
try:
    while True:
        print('coro yielded:', o.send(None))
except StopIteration:
    print('finished')