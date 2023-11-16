from __future__ import print_function
import itertools

def compiled_func(some_list):
    if False:
        return 10
    return some_list

def calledRepeatedly():
    if False:
        return 10
    compiled_f = compiled_func
    compiled_f(['some', 'random', 'values', 'to', 'check', 'call'])
    compiled_f([['some', 'other', 'values'], ['to', 'check', 'call']])
    pass
    return compiled_f
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')