"""The pure-python implementation of the StaticTuple type.

Note that it is generally just implemented as using tuples of tuples of
strings.
"""
from __future__ import absolute_import

class StaticTuple(tuple):
    """A static type, similar to a tuple of strings."""
    __slots__ = ()

    def __new__(cls, *args):
        if False:
            i = 10
            return i + 15
        if not args and _empty_tuple is not None:
            return _empty_tuple
        return tuple.__new__(cls, args)

    def __init__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        "Create a new 'StaticTuple'"
        num_keys = len(args)
        if num_keys < 0 or num_keys > 255:
            raise TypeError('StaticTuple(...) takes from 0 to 255 items')
        for bit in args:
            if type(bit) not in (str, StaticTuple, unicode, int, long, float, None.__class__, bool):
                raise TypeError('StaticTuple can only point to StaticTuple, str, unicode, int, long, float, bool, or None not %s' % (type(bit),))
        tuple.__init__(self)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '%s%s' % (self.__class__.__name__, tuple.__repr__(self))

    def __reduce__(self):
        if False:
            while True:
                i = 10
        return (StaticTuple, tuple(self))

    def __add__(self, other):
        if False:
            return 10
        'Concatenate self with other'
        return StaticTuple.from_sequence(tuple.__add__(self, other))

    def as_tuple(self):
        if False:
            print('Hello World!')
        return tuple(self)

    def intern(self):
        if False:
            for i in range(10):
                print('nop')
        return _interned_tuples.setdefault(self, self)

    @staticmethod
    def from_sequence(seq):
        if False:
            return 10
        'Convert a sequence object into a StaticTuple instance.'
        if isinstance(seq, StaticTuple):
            return seq
        return StaticTuple(*seq)
_empty_tuple = None
_empty_tuple = StaticTuple()
_interned_tuples = {}