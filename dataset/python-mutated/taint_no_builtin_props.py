def bad():
    if False:
        for i in range(10):
            print('nop')
    x = 'password'
    y = x
    z = y
    sink(z)

def ok1():
    if False:
        print('Hello World!')
    x = 'password'
    y = x
    z = y + 'ok'
    sink(z)

def ok2():
    if False:
        for i in range(10):
            print('nop')
    x = 'password'
    y = x
    z = f(y)
    sink(z)

def ok3():
    if False:
        print('Hello World!')
    x = 'password'
    y = x
    z = a[y]
    sink(z)