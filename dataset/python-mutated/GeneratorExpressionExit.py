import itertools

def calledRepeatedly():
    if False:
        i = 10
        return i + 15
    gen = (x for x in range(3))
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