from __future__ import print_function
a = 3
del a
try:
    del a
except NameError as e:
    print('Raised expected exception:', repr(e))

def someFunction(b, c):
    if False:
        return 10
    b = 1
    del b
    try:
        del b
    except UnboundLocalError as e:
        print('Raised expected exception:', repr(e))
someFunction(3, 4)