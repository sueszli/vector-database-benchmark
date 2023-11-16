@micropython.viper
def f():
    if False:
        while True:
            i = 10
    import micropython
    print(micropython.const(1))
    from micropython import const
    print(const(2))
f()