def ok():
    if False:
        print('Hello World!')
    x = source
    sanitize(x)
    sink(x)

def bad():
    if False:
        i = 10
        return i + 15
    x = source
    sink(x)

def also_bad():
    if False:
        i = 10
        return i + 15
    x = source
    sanitize(x)
    x = source
    sink(x)