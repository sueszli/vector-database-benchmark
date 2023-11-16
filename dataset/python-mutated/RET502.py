def x(y):
    if False:
        return 10
    if not y:
        return
    return 1

def x(y):
    if False:
        for i in range(10):
            print('nop')
    if not y:
        return
    return

def x(y):
    if False:
        while True:
            i = 10
    if y:
        return
    print()

def x(y):
    if False:
        return 10
    if not y:
        return 1
    return 2

def x(y):
    if False:
        return 10
    for i in range(10):
        if i > 100:
            return i
    else:
        return 1

def x(y):
    if False:
        while True:
            i = 10
    try:
        return 1
    except:
        return 2

def x(y):
    if False:
        while True:
            i = 10
    try:
        return 1
    finally:
        return 2

def x(y):
    if False:
        while True:
            i = 10
    if not y:
        return 1

    def inner():
        if False:
            while True:
                i = 10
        return
    return 2

def x(y):
    if False:
        for i in range(10):
            print('nop')
    if not y:
        return

    def inner():
        if False:
            return 10
        return 1