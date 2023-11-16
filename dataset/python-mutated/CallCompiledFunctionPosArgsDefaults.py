from __future__ import print_function
import itertools

def compiled_func(a=1, b=2, c=3, d=4, e=5, f=6):
    if False:
        return 10
    return (a, b, c, d, e, f)

def calledRepeatedly():
    if False:
        while True:
            i = 10
    compiled_f = compiled_func
    compiled_f()
    compiled_f()
    compiled_f()
    pass
    return compiled_f
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')