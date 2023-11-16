from __future__ import print_function
import sys

def eight():
    if False:
        for i in range(10):
            print('nop')
    return 8

def nine():
    if False:
        for i in range(10):
            print('nop')
    return 9

def raisy1():
    if False:
        return 10
    return 1 / 0

def raisy2():
    if False:
        for i in range(10):
            print('nop')
    return 1()

def raisy3(arg):
    if False:
        while True:
            i = 10
    raise TypeError(arg)

def returnInTried(for_call):
    if False:
        print('Hello World!')
    try:
        print('returnInTried with exception info in tried block:', sys.exc_info())
        return for_call()
    finally:
        print('returnInTried with exception info in final block:', sys.exc_info())

def returnInFinally(for_call):
    if False:
        for i in range(10):
            print('nop')
    try:
        print('returnInFinally with exception info in tried block:', sys.exc_info())
    finally:
        print('returnInFinally with exception info in final block:', sys.exc_info())
        return for_call()
print('Standard try finally with return in tried block:')
print('result', returnInTried(eight))
print('*' * 80)
print('Standard try finally with return in final block:')
print('result', returnInFinally(nine))
print('*' * 80)
print('Exception raising try finally with return in tried block:')
try:
    print('result', returnInTried(raisy1))
except Exception as e:
    print('Gave exception', repr(e))
print('*' * 80)
print('Exception raising try finally with return in final block:')
try:
    print('result', returnInFinally(raisy2))
except Exception as e:
    print('Gave exception', repr(e))
print('*' * 80)
try:
    raisy3('unreal 1')
except Exception as outer_e:
    print('Exception raising try finally with return in tried block:')
    try:
        print('result', returnInTried(raisy1))
    except Exception as e:
        print('Gave exception', repr(e))
    print('Handler exception remains:', repr(outer_e))
print('*' * 80)
try:
    raisy3('unreal 2')
except Exception as outer_e:
    print('Exception raising try finally with return in final block:')
    try:
        print('result', returnInFinally(raisy2))
    except Exception as e:
        print('Gave exception', repr(e))
    print('Handler exception remains:', repr(outer_e))
print('*' * 80)