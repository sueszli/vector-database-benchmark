def foo():
    if False:
        for i in range(10):
            print('nop')
    bar()

def bar():
    if False:
        i = 10
        return i + 15
    pass
foo()