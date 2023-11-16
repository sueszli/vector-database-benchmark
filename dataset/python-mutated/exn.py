def foo():
    if False:
        for i in range(10):
            print('nop')
    try:
        foo()
    except Exn as e:
        return e