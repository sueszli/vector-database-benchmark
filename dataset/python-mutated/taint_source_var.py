def bad():
    if False:
        i = 10
        return i + 15
    x = set([])
    taint(x)
    y = x
    sink(y)

def ok():
    if False:
        while True:
            i = 10
    x = set([])
    taint(x)
    x = set([])
    sink(y)