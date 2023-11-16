def gen():
    if False:
        for i in range(10):
            print('nop')
    print('sent:', (yield 1))
    yield 2

def gen2():
    if False:
        for i in range(10):
            print('nop')
    print((yield from gen()))
g = gen2()
next(g)
print('yielded:', g.send('val'))
try:
    next(g)
except StopIteration:
    print('StopIteration')