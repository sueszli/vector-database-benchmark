from __future__ import print_function
import sys
x = 2

def someFunction1():
    if False:
        for i in range(10):
            print('nop')
    x = 3
    return x

def someFunction2():
    if False:
        i = 10
        return i + 15
    global x
    x = 4
    return x

def someFunction3():
    if False:
        return 10
    return x

def someNestedGlobalUser1():
    if False:
        i = 10
        return i + 15
    z = 1

    def setZ():
        if False:
            return 10
        global z
        z = 3
    setZ()
    return z

def someNestedGlobalUser2():
    if False:
        print('Hello World!')
    z = 1
    exec('\ndef setZ():\n    global z\n\n    z = 3\n\nsetZ()\n')
    return z

def someNestedGlobalUser3a():
    if False:
        while True:
            i = 10
    exec('\nz = 1\n\ndef setZ():\n    global z\n\n    z = 3\n\nsetZ()\n')
    return (z, locals().keys() == ['setZ'])

def someNestedGlobalUser3b():
    if False:
        for i in range(10):
            print('nop')
    exec('\nz = 1\n')
    if sys.version_info[0] < 3:
        return (z, locals().keys() == ['z'])
    else:
        return locals().keys() == []

def someNestedGlobalUser4():
    if False:
        return 10
    z = 1
    exec('\nz = 2\n\ndef setZ():\n    global z\n\n    z = 3*z\n\nsetZ()\n')
    return z

def someNestedGlobalUser5():
    if False:
        while True:
            i = 10
    z = 1
    exec('\nz = 3\n\n')
    return z

def someNestedGlobalUser6():
    if False:
        for i in range(10):
            print('nop')
    exec('\nz = 7\n\n')
    return z
print('Function that shadows a global variable with a local variable')
print(someFunction1())
print('Function that accesses and changes a global variable declared with a global statement')
print(someFunction2())
print('Function that uses a global variable')
print(someFunction3())
print('Functions that uses a global variable in a nested function in various ways:')
print(someNestedGlobalUser1, someNestedGlobalUser1())
del z
print(someNestedGlobalUser2, someNestedGlobalUser2())
del z
print(someNestedGlobalUser3a, someNestedGlobalUser3a())
del z
print(someNestedGlobalUser3b, someNestedGlobalUser3b())
print(someNestedGlobalUser4, (someNestedGlobalUser4(), z))
del z
print(someNestedGlobalUser5, someNestedGlobalUser5())
z = 9
print(someNestedGlobalUser6, (someNestedGlobalUser6(), z))
x = 7

def f():
    if False:
        for i in range(10):
            print('nop')
    x = 1

    def g():
        if False:
            for i in range(10):
                print('nop')
        global x

        def i():
            if False:
                while True:
                    i = 10

            def h():
                if False:
                    print('Hello World!')
                return x
            return h()
        return i()
    return g()
print(f())
global global_already
global_already = 1