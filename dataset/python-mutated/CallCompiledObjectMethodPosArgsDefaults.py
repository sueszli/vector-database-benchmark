from __future__ import print_function
import itertools

class C(object):

    def compiled_method(self, a=1, b=2, c=3, d=4, e=5, f=6):
        if False:
            print('Hello World!')
        return (a, b, c, d, e, f)

def calledRepeatedly():
    if False:
        print('Hello World!')
    inst = C()
    inst.compiled_method()
    inst.compiled_method()
    inst.compiled_method()
    pass
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')