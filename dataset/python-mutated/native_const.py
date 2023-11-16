@micropython.native
def f():
    if False:
        return 10
    return b'bytes'
print(f())

@micropython.native
def f():
    if False:
        return 10

    @micropython.native
    def g():
        if False:
            i = 10
            return i + 15
        return 123
    return g
print(f()())