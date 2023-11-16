from __future__ import print_function
import sys

def displayDict(d):
    if False:
        while True:
            i = 10
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
    if '__cached__' in d:
        d['__cached__'] = '<__cached__ removed>'
    if '__spec__' in d:
        d['__spec__'].loader = 'loader removed'
        d['__spec__'].origin = 'origin removed'
        d['__spec__'].submodule_search_locations = 'submodule_search_locations removed'
    if '__compiled__' in d:
        del d['__compiled__']
    import pprint
    return pprint.pformat(d)
print(displayDict(globals()))
print('__name__: ', __name__)
print('__package__: ', __package__ if __package__ is not None or sys.version_info[:2] != (3, 2) else '.'.join(__name__.split('.')[:-1]))