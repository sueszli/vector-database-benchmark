from execme import dangerous

def foo(arg):
    if False:
        for i in range(10):
            print('nop')
    some_val = source()
    print('abc')
    dangerous(1, arg)