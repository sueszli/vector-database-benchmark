def test():
    if False:
        i = 10
        return i + 15
    x = set([])
    if cond:
        taint(x)
    y = x
    sink(y)