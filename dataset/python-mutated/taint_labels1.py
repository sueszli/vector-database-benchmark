def foo():
    if False:
        while True:
            i = 10
    a = source()
    b = a
    sink(b)

def bar(x):
    if False:
        return 10
    a = source()
    b = a + x
    sink(b)