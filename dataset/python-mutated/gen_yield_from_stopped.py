def gen():
    if False:
        i = 10
        return i + 15
    return 1
    yield
f = gen()

def run():
    if False:
        return 10
    print((yield from f))
    print((yield from f))
    print((yield from f))
try:
    next(run())
except StopIteration:
    print('StopIteration')

def run():
    if False:
        i = 10
        return i + 15
    print((yield from f))
f = zip()
try:
    next(run())
except StopIteration:
    print('StopIteration')