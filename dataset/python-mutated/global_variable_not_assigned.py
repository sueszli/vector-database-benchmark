def f():
    if False:
        print('Hello World!')
    global X

def f():
    if False:
        for i in range(10):
            print('nop')
    global X
    print(X)

def f():
    if False:
        while True:
            i = 10
    global X
    X = 1

def f():
    if False:
        while True:
            i = 10
    global X
    (X, y) = (1, 2)

def f():
    if False:
        return 10
    global X
    del X

def f():
    if False:
        return 10
    global X
    X += 1