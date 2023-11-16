from __future__ import print_function

def decorator1(f):
    if False:
        i = 10
        return i + 15
    print('Executing decorator 1')

    def deco_f():
        if False:
            while True:
                i = 10
        return f() + 2
    return deco_f

def decorator2(f):
    if False:
        return 10
    print('Executing decorator 2')

    def deco_f():
        if False:
            while True:
                i = 10
        return f() * 2
    return deco_f

@decorator1
@decorator2
def function1():
    if False:
        i = 10
        return i + 15
    return 3
print(function1())

def deco_returner1():
    if False:
        i = 10
        return i + 15
    print('Executing decorator returner D1')
    return decorator1

def deco_returner2():
    if False:
        while True:
            i = 10
    print('Executing decorator returner D2')
    return decorator2

@deco_returner1()
@deco_returner2()
def function2():
    if False:
        print('Hello World!')
    return 3
print(function2())

def function3():
    if False:
        while True:
            i = 10
    return 3
function3 = deco_returner1()(deco_returner2()(function3))
print(function3())