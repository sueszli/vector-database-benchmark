from __future__ import print_function
'Some doc'

def one():
    if False:
        while True:
            i = 10
    return 1

def tryScope1(x):
    if False:
        for i in range(10):
            print('nop')
    try:
        try:
            x += one()
        finally:
            print('Finally is executed')
            try:
                _z = one()
            finally:
                print('Deep Nested finally is executed')
    except:
        print('Exception occurred')
    else:
        print('No exception occurred')
tryScope1(1)
print('*' * 20)
tryScope1([1])

def tryScope2(x, someExceptionClass):
    if False:
        while True:
            i = 10
    try:
        x += 1
    except someExceptionClass as e:
        print('Exception class from argument occurred:', someExceptionClass, repr(e))
    else:
        print('No exception occurred')

def tryScope3(x):
    if False:
        print('Hello World!')
    if x:
        try:
            x += 1
        except TypeError:
            print('TypeError occurred')
    else:
        print('Not taken')
print('*' * 20)
tryScope2(1, TypeError)
tryScope2([1], TypeError)
print('*' * 20)
tryScope3(1)
tryScope3([1])
tryScope3([])
print('*' * 20)

def tryScope4(x):
    if False:
        i = 10
        return i + 15
    try:
        x += 1
    except:
        print('exception occurred')
    else:
        print('no exception occurred')
    finally:
        print('finally obeyed')
tryScope4(1)
tryScope4([1])

def tryScope5():
    if False:
        return 10
    import sys
    print('Exception info is initially', sys.exc_info())
    try:
        try:
            undefined_global += 1
        finally:
            print("Exception info in 'finally' clause is", sys.exc_info())
    except:
        pass
tryScope5()