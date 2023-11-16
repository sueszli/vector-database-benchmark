import itertools

def calledRepeatedly():
    if False:
        i = 10
        return i + 15

    def generator():
        if False:
            i = 10
            return i + 15
        yield 1
        yield 2
        yield 3
    gen = generator()
    next(gen)
    throw = gen.throw
    exc = GeneratorExit
    try:
        throw(exc)
        pass
    except exc:
        pass
    return (throw, exc)
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')