def foo(x):
    if False:
        while True:
            i = 10
    return x

@foo
def bar():
    if False:
        while True:
            i = 10
    pass

def bar():
    if False:
        for i in range(10):
            print('nop')
    pass