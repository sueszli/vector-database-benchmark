from __future__ import print_function
import itertools

class C(object):

    def compiled_method(self):
        if False:
            print('Hello World!')
        return self

def calledRepeatedly():
    if False:
        i = 10
        return i + 15
    inst = C()
    inst.compiled_method()
    inst.compiled_method()
    inst.compiled_method()
    pass
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')