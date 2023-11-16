from __future__ import print_function
import itertools

class C(object):

    def compiled_method(self, a, b, c, d, e, f):
        if False:
            return 10
        return (a, b, c, d, e, f)

def calledRepeatedly():
    if False:
        while True:
            i = 10
    inst = C()
    inst.compiled_method('some', 'random', 'values', 'to', 'check', 'call')
    inst.compiled_method('some', 'other', 'values', 'to', 'check', 'call')
    inst.compiled_method('some', 'new', 'values', 'to', 'check', 'call')
    pass
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')