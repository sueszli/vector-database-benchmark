""" Test that covers all module attributes. """
from __future__ import print_function
import package_level1.Nearby1
import package_level1.package_level2.Nearby2
import package_level1.package_level2.package_level3
import package_level1.package_level2.package_level3.Nearby3

def displayDict(d):
    if False:
        i = 10
        return i + 15
    d = dict(d)
    del d['displayDict']
    del d['__builtins__']
    if '__loader__' in d:
        if str is bytes:
            del d['__loader__']
        else:
            d['__loader__'] = '<__loader__ removed>'
    if '__file__' in d:
        d['__file__'] = '<__file__ removed>'
    if '__compiled__' in d:
        del d['__compiled__']
    import pprint
    return pprint.pformat(d)
print(displayDict(globals()))