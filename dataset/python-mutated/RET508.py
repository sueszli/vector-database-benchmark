def foo2(x, y, w, z):
    if False:
        print('Hello World!')
    for i in x:
        if i > y:
            break
        elif i > w:
            break
        else:
            a = z

def foo5(x, y, z):
    if False:
        for i in range(10):
            print('nop')
    for i in x:
        if i > y:
            if y:
                a = 4
            else:
                b = 2
            break
        elif z:
            c = 2
        else:
            c = 3
        break

def foo1(x, y, z):
    if False:
        while True:
            i = 10
    for i in x:
        if i > y:
            break
        else:
            a = z

def foo3(x, y, z):
    if False:
        print('Hello World!')
    for i in x:
        if i > y:
            a = 1
            if z:
                b = 2
                break
            else:
                c = 3
                break
        else:
            d = 4
            break

def foo4(x, y):
    if False:
        i = 10
        return i + 15
    for i in x:
        if i > x:
            if y:
                a = 4
            else:
                b = 2
            break
        else:
            c = 3
        break

def foo6(x, y):
    if False:
        i = 10
        return i + 15
    for i in x:
        if i > x:
            if y:
                a = 4
                break
            else:
                b = 2
        else:
            c = 3
        break

def bar4(x):
    if False:
        for i in range(10):
            print('nop')
    for i in range(10):
        if x:
            break
        else:
            try:
                return
            except ValueError:
                break

def bar1(x, y, z):
    if False:
        i = 10
        return i + 15
    for i in x:
        if i > y:
            break
        return z

def bar2(w, x, y, z):
    if False:
        for i in range(10):
            print('nop')
    for i in w:
        if i > x:
            break
        if z < i:
            a = y
        else:
            break
        return

def bar3(x, y, z):
    if False:
        print('Hello World!')
    for i in x:
        if i > y:
            if z:
                break
        else:
            return z
        return None

def nested1(x, y, z):
    if False:
        print('Hello World!')
    for i in x:
        if i > x:
            for j in y:
                if j < z:
                    break
        else:
            a = z

def nested2(x, y, z):
    if False:
        i = 10
        return i + 15
    for i in x:
        if i > x:
            for j in y:
                if j > z:
                    break
                break
        else:
            a = z

def elif1(x, y, w, z):
    if False:
        for i in range(10):
            print('nop')
    for i in x:
        if i > y:
            a = z
        elif i > w:
            break
        else:
            a = z

def elif2(x, y, w, z):
    if False:
        i = 10
        return i + 15
    for i in x:
        if i > y:
            a = z
        elif i > w:
            break
        else:
            a = z