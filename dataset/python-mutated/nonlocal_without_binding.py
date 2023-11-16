nonlocal x

def f():
    if False:
        for i in range(10):
            print('nop')
    nonlocal x

def f():
    if False:
        while True:
            i = 10
    nonlocal y

def f():
    if False:
        print('Hello World!')
    x = 1

    def f():
        if False:
            while True:
                i = 10
        nonlocal x

    def f():
        if False:
            print('Hello World!')
        nonlocal y

def f():
    if False:
        while True:
            i = 10
    x = 1

    def g():
        if False:
            while True:
                i = 10
        nonlocal x
    del x

def f():
    if False:
        print('Hello World!')

    def g():
        if False:
            for i in range(10):
                print('nop')
        nonlocal x
    del x

def f():
    if False:
        for i in range(10):
            print('nop')
    try:
        pass
    except Exception as x:
        pass

    def g():
        if False:
            return 10
        nonlocal x
        x = 2