def foo():
    if False:
        for i in range(10):
            print('nop')
    a = source()
    if cond():
        b = a
    sink(b)