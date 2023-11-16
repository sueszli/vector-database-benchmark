def gen():
    if False:
        for i in range(10):
            print('nop')
    yield from gen()
try:
    list(gen())
except RuntimeError:
    print('RuntimeError')

def gen2():
    if False:
        return 10
    for x in gen2():
        yield x
try:
    next(gen2())
except RuntimeError:
    print('RuntimeError')