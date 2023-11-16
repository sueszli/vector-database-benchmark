from __future__ import print_function
import itertools
exec('\ndef python_func(a,b,c,d,e,f):\n    pass\n')

def calledRepeatedly():
    if False:
        while True:
            i = 10
    python_f = python_func
    python_f('some', 'random', 'values', 'to', 'check', 'call')
    python_f('some', 'other', 'values', 'to', 'check', 'call')
    python_f('some', 'new', 'values', 'to', 'check', 'call')
    pass
    return python_f
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')