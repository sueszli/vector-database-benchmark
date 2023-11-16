""" Some module documentation.

With newline and stuff."""
from __future__ import print_function
import os
import sys
print('doc:', __doc__)
print('filename:', os.path.basename(__file__))
print('builtins:', __builtins__)
print('debug', __debug__)
print('debug in builtins', __builtins__.__debug__)
print('__initializing__', end=' ')
try:
    print(__initializing__)
except NameError:
    print('not found')

def checkFromFunction():
    if False:
        for i in range(10):
            print('nop')
    frame = sys._getframe(1)

    def displayDict(d):
        if False:
            for i in range(10):
                print('nop')
        if '__loader__' in d:
            d = dict(d)
            if str is bytes:
                del d['__loader__']
            else:
                d['__loader__'] = '<__loader__ removed>'
        if '__file__' in d:
            d = dict(d)
            d['__file__'] = '<__file__ removed>'
        if '__compiled__' in d:
            d = dict(d)
            del d['__compiled__']
        import pprint
        return pprint.pformat(d)
    print('Globals', displayDict(frame.f_globals))
    print('Locals', displayDict(frame.f_locals))
    print('Is identical', frame.f_locals is frame.f_globals)
checkFromFunction()