from __future__ import print_function
import itertools

def compiled_func(a, b, c, d, e, f):
    if False:
        while True:
            i = 10
    return (a, b, c, d, e, f)

def calledRepeatedly():
    if False:
        for i in range(10):
            print('nop')
    compiled_f = compiled_func
    compiled_f('some', 'random', 'values', 'to', 'check', 'call')
    compiled_f('some', 'other', 'values', 'to', 'check', 'call')
    compiled_f('some', 'new', 'values', 'to', 'check', 'call')
    pass
    return compiled_f
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')