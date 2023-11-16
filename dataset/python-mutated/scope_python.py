def foo(x):
    if False:
        return 10
    return 1

def foo(x):
    if False:
        while True:
            i = 10
    return x

def bar(x):
    if False:
        for i in range(10):
            print('nop')
    return x

def foo(x):
    if False:
        return 10

    def bar(x):
        if False:
            return 10
        return x
    yield bar(x)
glob = 1

def foo(x):
    if False:
        print('Hello World!')
    global glob
    return complex_call(glob)

def foo2(x):
    if False:
        i = 10
        return i + 15
    global glob
    return complex_call(glob)

def foo3(x):
    if False:
        while True:
            i = 10
    global glob
    return complex_call_different(glob)

def foo(x):
    if False:
        while True:
            i = 10
    return complex_call(x)

def foo2(x):
    if False:
        print('Hello World!')
    return complex_call(x)