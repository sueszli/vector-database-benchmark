def test():
    if False:
        i = 10
        return i + 15
    x = source()
    x.a = sanitize()
    x.a.i = source()
    sink(x.a.i)
    sink(x.a)
    sink(x)