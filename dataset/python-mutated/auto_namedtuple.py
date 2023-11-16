from __future__ import annotations
import collections

def auto_namedtuple(classname='auto_namedtuple', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Returns an automatic namedtuple object.\n\n    Args:\n        classname - The class name for the returned object.\n        **kwargs - Properties to give the returned object.\n    '
    return collections.namedtuple(classname, kwargs.keys())(**kwargs)