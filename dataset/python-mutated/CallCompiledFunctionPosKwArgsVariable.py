from __future__ import print_function
import itertools

def compiled_func(a, b, c, d, e, f):
    if False:
        for i in range(10):
            print('nop')
    return (a, b, c, d, e, f)

def getUnknownValue():
    if False:
        while True:
            i = 10
    return 8

def calledRepeatedly():
    if False:
        i = 10
        return i + 15
    a = getUnknownValue()
    b = getUnknownValue()
    c = getUnknownValue()
    d = getUnknownValue()
    e = getUnknownValue()
    f = getUnknownValue()
    compiled_f = compiled_func
    compiled_f(a, b, c, d=d, e=e, f=f)
    compiled_f(a, c, b, d=d, e=e, f=f)
    compiled_f(a, b, c, d=d, e=e, f=f)
    pass
    return compiled_f
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')