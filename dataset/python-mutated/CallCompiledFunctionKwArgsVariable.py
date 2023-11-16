from __future__ import print_function
import itertools

def compiled_func(a, b, c):
    if False:
        return 10
    return (a, b, c)

def getUnknownValue():
    if False:
        print('Hello World!')
    return 8

def calledRepeatedly():
    if False:
        for i in range(10):
            print('nop')
    a = getUnknownValue()
    b = getUnknownValue()
    c = getUnknownValue()
    compiled_f = compiled_func
    compiled_f(a=a, b=b, c=c)
    compiled_f(a=a, b=b, c=c)
    compiled_f(a=a, b=b, c=c)
    pass
    return compiled_f
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')