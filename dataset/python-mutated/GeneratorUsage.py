import itertools

def calledRepeatedly():
    if False:
        return 10

    def generator():
        if False:
            while True:
                i = 10
        yield 1
        yield 2
        yield 3
    gen = generator()
    x = next(gen)
    next(gen)
    return x
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')