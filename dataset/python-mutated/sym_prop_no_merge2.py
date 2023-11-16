def test1():
    if False:
        return 10
    if cond():
        y = f(x)
        z = g(y)
    return z

def test2():
    if False:
        return 10
    while cond():
        y = f(x)
        z = g(y)
    return z