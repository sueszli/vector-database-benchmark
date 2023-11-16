from __future__ import print_function

def closureTest1():
    if False:
        print('Hello World!')
    d = 1

    def subby():
        if False:
            i = 10
            return i + 15
        return d
    d = 22222 * 2222
    return subby()

def closureTest2():
    if False:
        print('Hello World!')

    def subby():
        if False:
            for i in range(10):
                print('nop')
        return d
    d = 2222 * 2222
    return subby()

def closureTest3():
    if False:
        i = 10
        return i + 15

    def subby():
        if False:
            return 10
        return undefined_global
    try:
        return subby()
    except NameError:
        return 88
d = 1

def scopeTest4():
    if False:
        return 10
    try:
        return d
        d = 1
    except UnboundLocalError as e:
        return repr(e)
print('Test closure where value is overwritten:', closureTest1())
print('Test closure where value is assigned only late:', closureTest2())
print('Test function where closured value is never assigned:', closureTest3())
print('Scope test where UnboundLocalError is expected:', scopeTest4())

def function():
    if False:
        for i in range(10):
            print('nop')
    pass

class ClosureLocalizerClass:
    print('Function before assigned in a class:', function)
    function = 1
    print('Function after it was assigned in class:', function)
ClosureLocalizerClass()

def ClosureLocalizerFunction():
    if False:
        i = 10
        return i + 15
    try:
        function = function
        print("Function didn't give unbound local error")
    except UnboundLocalError as e:
        print('Function gave unbound local error when accessing function before assignment:', repr(e))
ClosureLocalizerFunction()

class X:

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.x = x

def changingClosure():
    if False:
        while True:
            i = 10
    print('Changing a closure taken value after it was taken.')
    a = 1

    def closureTaker():
        if False:
            while True:
                i = 10
        return X(a)
    x = closureTaker()
    a = 2
    print('Closure value first time:', x.x)
    x = closureTaker()
    print('Closure value second time:', x.x)
changingClosure()