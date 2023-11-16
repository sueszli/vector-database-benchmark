def foo():
    if False:
        return 10
    foo.Bar.alice = 2 * my_module.UNIT

def bar():
    if False:
        print('Hello World!')
    x = foo.Bar
    x.alice = 3 * my_module.UNIT

def alice():
    if False:
        i = 10
        return i + 15
    x = foo.Bar
    y = x
    y.alice = 4 * my_module.UNIT