@micropython.viper
def f():
    if False:
        i = 10
        return i + 15
    return b'bytes'
print(f())

@micropython.viper
def f():
    if False:
        while True:
            i = 10

    @micropython.viper
    def g() -> int:
        if False:
            while True:
                i = 10
        return 123
    return g
print(f()())