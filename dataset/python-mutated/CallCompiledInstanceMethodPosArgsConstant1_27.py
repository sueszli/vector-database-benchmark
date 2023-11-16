from __future__ import print_function
import itertools

class C:

    def compiled_method(self, a):
        if False:
            print('Hello World!')
        return a

def calledRepeatedly():
    if False:
        print('Hello World!')
    inst = C()
    inst.compiled_method('some')
    inst.compiled_method('some')
    inst.compiled_method('some')
    pass
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')