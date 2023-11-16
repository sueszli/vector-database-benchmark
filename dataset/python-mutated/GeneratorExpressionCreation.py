import itertools

def calledRepeatedly():
    if False:
        print('Hello World!')
    gen = (x for x in range(3))
    gen = iter((1, 2, 3))
    x = next(gen)
    next(gen)
    return x
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')