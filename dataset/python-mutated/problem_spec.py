"""Wrapper around a training problem."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple

class Spec(namedtuple('Spec', 'callable args kwargs')):
    """Syntactic sugar for keeping track of a function/class + args."""
    __slots__ = ()

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the output of the callable.'
        return self.callable(*self.args, **self.kwargs)