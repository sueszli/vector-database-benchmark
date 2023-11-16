def no_precondition_for_if(x):
    if False:
        return 10
    if x:
        True
    else:
        False

def no_precondition_for_condition(x):
    if False:
        i = 10
        return i + 15
    return 0 if x else 1

def some_source(name):
    if False:
        for i in range(10):
            print('nop')
    pass

def issue1():
    if False:
        i = 10
        return i + 15
    x = some_source('my-data')
    if x:
        return

def issue2():
    if False:
        while True:
            i = 10
    x = some_source('other')
    return 0 if x else 1