from __future__ import print_function
import itertools

class C(object):

    def compiled_method(self, a, b, c, d, e, f):
        if False:
            for i in range(10):
                print('nop')
        return (a, b, c, d, e, f)

def getUnknownValue():
    if False:
        return 10
    return 8

def calledRepeatedly():
    if False:
        i = 10
        return i + 15
    a = getUnknownValue()
    b = getUnknownValue()
    c = getUnknownValue()
    d = getUnknownValue()
    e = getUnknownValue()
    f = getUnknownValue()
    inst = C()
    inst.compiled_method(a, b, c, d, e, f)
    inst.compiled_method(a, c, b, d, e, f)
    inst.compiled_method(a, b, c, d, f, e)
    pass
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')