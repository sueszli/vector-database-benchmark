def test(x):
    if False:
        i = 10
        return i + 15
    c = 42
    if x < 0:
        yield c
    else:
        yield x