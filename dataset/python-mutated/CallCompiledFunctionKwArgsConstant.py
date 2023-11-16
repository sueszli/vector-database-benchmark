from __future__ import print_function
import itertools

def compiled_func(a, b, c):
    if False:
        return 10
    return (a, b, c)

def calledRepeatedly():
    if False:
        i = 10
        return i + 15
    compiled_f = compiled_func
    compiled_f(a='some', b='random', c='values')
    compiled_f(a='some', b='other', c='values')
    compiled_f(a='some', b='new', c='value set')
    pass
    return compiled_f
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')