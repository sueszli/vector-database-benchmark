def foo():
    if False:
        for i in range(10):
            print('nop')
    if x > 2:
        foo()
        bar()
    else:
        foo()
        bar()
    if x > 2:
        foo()
        bar()
    else:
        foo()