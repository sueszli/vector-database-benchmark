def test(c):
    if False:
        return 10
    if c:
        x = a
        y = x.foo
    else:
        x = b
        y = x.foo
    return y