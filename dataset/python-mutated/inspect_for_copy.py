"""A version of inspect that includes what 'copy' needs.

Importing the python standard module 'copy' is far more expensive than it
needs to be, because copy imports 'inspect' which imports 'tokenize'.
And 'copy' only needs 2 small functions out of 'inspect', but has to
load all of 'tokenize', which makes it horribly slow.

This module is designed to use tricky hacks in import rules, to avoid this
overhead.
"""
from __future__ import absolute_import

def _searchbases(cls, accum):
    if False:
        return 10
    if cls in accum:
        return
    accum.append(cls)
    for base in cls.__bases__:
        _searchbases(base, accum)

def getmro(cls):
    if False:
        for i in range(10):
            print('nop')
    'Return tuple of base classes (including cls) in method resolution order.'
    if hasattr(cls, '__mro__'):
        return cls.__mro__
    else:
        result = []
        _searchbases(cls, result)
        return tuple(result)

def import_copy_with_hacked_inspect():
    if False:
        i = 10
        return i + 15
    "Import the 'copy' module with a hacked 'inspect' module"
    import sys
    if 'inspect' in sys.modules:
        import copy
        return
    mod = __import__('bzrlib.inspect_for_copy', globals(), locals(), ['getmro'])
    sys.modules['inspect'] = mod
    try:
        import copy
    finally:
        del sys.modules['inspect']