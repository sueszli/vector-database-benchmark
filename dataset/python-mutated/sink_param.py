def foo():
    if False:
        for i in range(10):
            print('nop')
    a = source1()
    b = 'safe'
    sink1(a, b)
    sink1(b, a)