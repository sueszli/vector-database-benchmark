l = [1, 2, 3, 4]

def foo():
    if False:
        while True:
            i = 10
    yield from l
    yield from l
    a = (yield from l)
    with (yield from l):
        pass
    c = [(yield from l), (yield from l)]
    while (yield from l):
        pass
    yield (yield from l)