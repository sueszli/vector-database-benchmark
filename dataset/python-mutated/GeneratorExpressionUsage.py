import itertools
not_all = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]

def calledRepeatedly():
    if False:
        i = 10
        return i + 15
    gen = (x for x in not_all)
    x = next(gen)
    all(gen)
    y = next(gen)
    return (x, y)
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')