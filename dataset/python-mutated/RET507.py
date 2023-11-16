def foo2(x, y, w, z):
    if False:
        return 10
    for i in x:
        if i < y:
            continue
        elif i < w:
            continue
        else:
            a = z

def foo5(x, y, z):
    if False:
        print('Hello World!')
    for i in x:
        if i < y:
            if y:
                a = 4
            else:
                b = 2
            continue
        elif z:
            c = 2
        else:
            c = 3
        continue

def foo1(x, y, z):
    if False:
        for i in range(10):
            print('nop')
    for i in x:
        if i < y:
            continue
        else:
            a = z

def foo3(x, y, z):
    if False:
        print('Hello World!')
    for i in x:
        if i < y:
            a = 1
            if z:
                b = 2
                continue
            else:
                c = 3
                continue
        else:
            d = 4
            continue

def foo4(x, y):
    if False:
        for i in range(10):
            print('nop')
    for i in x:
        if i < x:
            if y:
                a = 4
            else:
                b = 2
            continue
        else:
            c = 3
        continue

def foo6(x, y):
    if False:
        while True:
            i = 10
    for i in x:
        if i < x:
            if y:
                a = 4
                continue
            else:
                b = 2
        else:
            c = 3
        continue

def bar4(x):
    if False:
        i = 10
        return i + 15
    for i in range(10):
        if x:
            continue
        else:
            try:
                return
            except ValueError:
                continue

def bar1(x, y, z):
    if False:
        while True:
            i = 10
    for i in x:
        if i < y:
            continue
        return z

def bar3(x, y, z):
    if False:
        for i in range(10):
            print('nop')
    for i in x:
        if i < y:
            if z:
                continue
        else:
            return z
        return None