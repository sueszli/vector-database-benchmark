def ok1():
    if False:
        print('Hello World!')
    x = 'tainted'
    y = x == 'safe'
    sink(y)

def ok2():
    if False:
        i = 10
        return i + 15
    x = 'tainted'
    sink(x == 'safe')

def ok3():
    if False:
        i = 10
        return i + 15
    x = 'tainted'
    y = 'something ' + x
    sink(x != 'safe')

def bad():
    if False:
        return 10
    x = 'tainted'
    y = x or 'safe'
    sink(y)