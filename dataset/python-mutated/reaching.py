def foo():
    if False:
        for i in range(10):
            print('nop')
    a = 1
    if True:
        a = 2
    else:
        a = 3
    print(a)