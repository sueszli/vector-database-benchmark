from __future__ import print_function
import itertools
exec('\ndef python_func(a,b,c,d,e,f):\n    pass\n')
a = (1, 2, 3, 4, 5)

def calledRepeatedly(python_f):
    if False:
        i = 10
        return i + 15
    args = a
    python_f(3, *args)
    pass
    return (python_f, args)
for x in itertools.repeat(None, 50000):
    calledRepeatedly(python_func)
print('OK.')