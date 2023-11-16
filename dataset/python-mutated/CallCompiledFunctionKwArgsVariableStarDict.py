from __future__ import print_function
import itertools

def compiled_func(a, b, c, d, e, f):
    if False:
        print('Hello World!')
    return (a, b, c, d, e, f)

def getUnknownValue():
    if False:
        for i in range(10):
            print('nop')
    return 8
arg_dict = {'d': 9, 'e': 9, 'f': 9}

def calledRepeatedly():
    if False:
        while True:
            i = 10
    a = getUnknownValue()
    b = getUnknownValue()
    c = getUnknownValue()
    compiled_f = compiled_func
    compiled_f(a=a, b=b, c=c, **arg_dict)
    compiled_f(a=a, b=b, c=c, **arg_dict)
    compiled_f(a=a, b=b, c=c, **arg_dict)
    pass
    return compiled_f
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')