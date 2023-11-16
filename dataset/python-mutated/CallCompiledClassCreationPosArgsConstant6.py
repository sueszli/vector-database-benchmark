from __future__ import print_function
import itertools

class C(object):

    def __init__(self, a, b, c, d, e, f):
        if False:
            for i in range(10):
                print('nop')
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

def calledRepeatedly():
    if False:
        print('Hello World!')
    C
    C('some', 'random', 'values', 'to', 'check', 'call')
    C('some', 'other', 'values', 'to', 'check', 'call')
    C('some', 'new', 'values', 'to', 'check', 'call')
    pass
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')