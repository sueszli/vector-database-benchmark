from __future__ import print_function
import itertools

class C(object):

    def compiled_method(self, a):
        if False:
            for i in range(10):
                print('nop')
        return a

def calledRepeatedly():
    if False:
        return 10
    inst = C()
    inst.compiled_method('some')
    inst.compiled_method('some')
    inst.compiled_method('some')
    pass
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')