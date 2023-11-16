def foo2(x, y, w, z):
    if False:
        i = 10
        return i + 15
    if x:
        a = 1
        return y
    elif z:
        b = 2
        return w
    else:
        c = 3
        return z

def foo5(x, y, z):
    if False:
        print('Hello World!')
    if x:
        if y:
            a = 4
        else:
            b = 2
        return
    elif z:
        c = 2
    else:
        c = 3
    return

def foo2(x, y, w, z):
    if False:
        return 10
    if x:
        a = 1
    elif z:
        b = 2
    else:
        c = 3
    if x:
        a = 1
        return y
    elif z:
        b = 2
        return w
    else:
        c = 3
        return z

def foo1(x, y, z):
    if False:
        i = 10
        return i + 15
    if x:
        a = 1
        return y
    else:
        b = 2
        return z

def foo3(x, y, z):
    if False:
        while True:
            i = 10
    if x:
        a = 1
        if y:
            b = 2
            return y
        else:
            c = 3
            return x
    else:
        d = 4
        return z

def foo4(x, y):
    if False:
        while True:
            i = 10
    if x:
        if y:
            a = 4
        else:
            b = 2
        return
    else:
        c = 3
    return

def foo6(x, y):
    if False:
        for i in range(10):
            print('nop')
    if x:
        if y:
            a = 4
            return
        else:
            b = 2
    else:
        c = 3
    return

def bar4(x):
    if False:
        return 10
    if x:
        return True
    else:
        try:
            return False
        except ValueError:
            return None

def bar1(x, y, z):
    if False:
        i = 10
        return i + 15
    if x:
        return y
    return z

def bar2(w, x, y, z):
    if False:
        i = 10
        return i + 15
    if x:
        return y
    if z:
        a = 1
    else:
        return w
    return None

def bar3(x, y, z):
    if False:
        print('Hello World!')
    if x:
        if z:
            return y
    else:
        return z
    return None
x = 0
if x == 1:
    y = 'a'
elif x == 2:
    y = 'b'
else:
    y = 'c'