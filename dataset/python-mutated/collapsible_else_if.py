"""
Test for else-if-used
"""

def ok0():
    if False:
        for i in range(10):
            print('nop')
    'Should not trigger on elif'
    if 1:
        pass
    elif 2:
        pass

def ok1():
    if False:
        return 10
    "If the orelse has more than 1 item in it, shouldn't trigger"
    if 1:
        pass
    else:
        print()
        if 1:
            pass

def ok2():
    if False:
        for i in range(10):
            print('nop')
    "If the orelse has more than 1 item in it, shouldn't trigger"
    if 1:
        pass
    else:
        if 1:
            pass
        print()

def not_ok0():
    if False:
        while True:
            i = 10
    if 1:
        pass
    elif 2:
        pass

def not_ok1():
    if False:
        i = 10
        return i + 15
    if 1:
        pass
    elif 2:
        pass
    else:
        pass

def not_ok2():
    if False:
        print('Hello World!')
    if True:
        print(1)
    elif True:
        print(2)
    elif True:
        print(3)
    else:
        print(4)