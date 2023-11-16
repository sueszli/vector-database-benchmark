from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
from typing import Optional
import __future__
import marshal
import random
import weakref
import re
import sys
import time
import io
import types
import copyreg
import functools
import renpy
FUTURE_FLAGS = __future__.CO_FUTURE_DIVISION | __future__.CO_FUTURE_WITH_STATEMENT

def _reconstructor(cls, base, state):
    if False:
        return 10
    if cls is RevertableSet and base is object:
        base = set
        state = []
    if base is object:
        obj = object.__new__(cls)
    else:
        obj = base.__new__(cls, state)
        if base.__init__ != object.__init__:
            base.__init__(obj, state)
    return obj
copyreg._reconstructor = _reconstructor
mutate_flag = True
if PY2:

    def _method_wrapper(method):
        if False:
            for i in range(10):
                print('nop')
        return functools.wraps(method, ('__name__', '__doc__'), ())
else:
    _method_wrapper = functools.wraps

def mutator(method):
    if False:
        while True:
            i = 10

    @_method_wrapper(method)
    def do_mutation(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        global mutate_flag
        mutated = renpy.game.log.mutated
        if id(self) not in mutated:
            mutated[id(self)] = (weakref.ref(self), self._clean())
            mutate_flag = True
        return method(self, *args, **kwargs)
    return do_mutation

class CompressedList(object):
    """
    Compresses the changes in a queue-like list. What this does is to try
    to find a central sub-list for which has objects in both lists. It
    stores the location of that in the new list, and then elements before
    and after in the sub-list.

    This only really works if the objects in the list are unique, but the
    results are efficient even if this doesn't work.
    """

    def __init__(self, old, new):
        if False:
            for i in range(10):
                print('nop')
        new_center = (len(new) - 1) // 2
        new_pivot = new[new_center]
        old_half = (len(old) - 1) // 2
        for i in range(0, old_half + 1):
            if old[old_half - i] is new_pivot:
                old_center = old_half - i
                break
            if old[old_half + i] is new_pivot:
                old_center = old_half + i
                break
        else:
            self.pre = old
            self.start = 0
            self.end = 0
            self.post = []
            return
        new_start = new_center
        new_end = new_center + 1
        old_start = old_center
        old_end = old_center + 1
        len_new = len(new)
        len_old = len(old)
        while new_start and old_start and (new[new_start - 1] is old[old_start - 1]):
            new_start -= 1
            old_start -= 1
        while new_end < len_new and old_end < len_old and (new[new_end] is old[old_end]):
            new_end += 1
            old_end += 1
        self.pre = list.__getitem__(old, slice(0, old_start))
        self.start = new_start
        self.end = new_end
        self.post = list.__getitem__(old, slice(old_end, len_old))

    def decompress(self, new):
        if False:
            return 10
        return self.pre + new[self.start:self.end] + self.post

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<CompressedList {} [{}:{}] {}>'.format(self.pre, self.start, self.end, self.post)

class RevertableList(list):

    def __init__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        log = renpy.game.log
        if log is not None:
            log.mutated[id(self)] = None
        list.__init__(self, *args)
    __delitem__ = mutator(list.__delitem__)
    if PY2:
        __delslice__ = mutator(list.__delslice__)
    __setitem__ = mutator(list.__setitem__)
    if PY2:
        __setslice__ = mutator(list.__setslice__)
    __iadd__ = mutator(list.__iadd__)
    __imul__ = mutator(list.__imul__)
    append = mutator(list.append)
    extend = mutator(list.extend)
    insert = mutator(list.insert)
    pop = mutator(list.pop)
    remove = mutator(list.remove)
    reverse = mutator(list.reverse)
    sort = mutator(list.sort)

    def wrapper(method):
        if False:
            while True:
                i = 10

        @_method_wrapper(method)
        def newmethod(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            l = method(*args, **kwargs)
            if l is NotImplemented:
                return l
            return RevertableList(l)
        return newmethod
    __add__ = wrapper(list.__add__)
    if PY2:
        __getslice__ = wrapper(list.__getslice__)
    __mul__ = wrapper(list.__mul__)
    __rmul__ = wrapper(list.__rmul__)
    del wrapper

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        rv = list.__getitem__(self, index)
        if isinstance(index, slice):
            return RevertableList(rv)
        else:
            return rv

    def copy(self):
        if False:
            print('Hello World!')
        return self[:]

    def clear(self):
        if False:
            i = 10
            return i + 15
        del self[:]

    def _clean(self):
        if False:
            print('Hello World!')
        '\n        Gets a clean copy of this object before any mutation occurs.\n        '
        return self[:]

    def _compress(self, clean):
        if False:
            i = 10
            return i + 15
        '\n        Takes a clean copy of this object, compresses it, and returns compressed\n        information that can be passed to rollback.\n        '
        if not self or not clean:
            return clean
        if renpy.config.list_compression_length is None:
            return clean
        if len(self) < renpy.config.list_compression_length or len(clean) < renpy.config.list_compression_length:
            return clean
        return CompressedList(clean, self)

    def _rollback(self, compressed):
        if False:
            print('Hello World!')
        '\n        Rolls this object back, using the information created by _compress.\n\n        Since compressed can come from a save file, this method also has to\n        recognize and deal with old data.\n        '
        if isinstance(compressed, CompressedList):
            self[:] = compressed.decompress(self)
        else:
            self[:] = compressed

def revertable_range(*args):
    if False:
        i = 10
        return i + 15
    return RevertableList(range(*args))

def revertable_sorted(*args, **kwargs):
    if False:
        print('Hello World!')
    return RevertableList(sorted(*args, **kwargs))

class RevertableDict(dict):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        log = renpy.game.log
        if log is not None:
            log.mutated[id(self)] = None
        dict.__init__(self, *args, **kwargs)
    __delitem__ = mutator(dict.__delitem__)
    __setitem__ = mutator(dict.__setitem__)
    clear = mutator(dict.clear)
    pop = mutator(dict.pop)
    popitem = mutator(dict.popitem)
    setdefault = mutator(dict.setdefault)
    update = mutator(dict.update)
    if PY2:

        def keys(self):
            if False:
                while True:
                    i = 10
            rv = dict.keys(self)
            if sys._getframe(1).f_code.co_flags & FUTURE_FLAGS != FUTURE_FLAGS:
                rv = RevertableList(rv)
            return rv

        def values(self):
            if False:
                for i in range(10):
                    print('nop')
            rv = dict.values(self)
            if sys._getframe(1).f_code.co_flags & FUTURE_FLAGS != FUTURE_FLAGS:
                rv = RevertableList(rv)
            return rv

        def items(self):
            if False:
                return 10
            rv = dict.items(self)
            if sys._getframe(1).f_code.co_flags & FUTURE_FLAGS != FUTURE_FLAGS:
                rv = RevertableList(rv)
            return rv
    else:
        itervalues = dict.values
        iterkeys = dict.keys
        iteritems = dict.items

        def has_key(self, key):
            if False:
                i = 10
                return i + 15
            return key in self

        def __or__(self, other):
            if False:
                i = 10
                return i + 15
            if not isinstance(other, dict):
                return NotImplemented
            rv = RevertableDict(self)
            rv.update(other)
            return rv

        def __ror__(self, other):
            if False:
                for i in range(10):
                    print('nop')
            if not isinstance(other, dict):
                return NotImplemented
            rv = RevertableDict(other)
            rv.update(self)
            return rv

        def __ior__(self, other):
            if False:
                print('Hello World!')
            self.update(other)
            return self

    def copy(self):
        if False:
            print('Hello World!')
        rv = RevertableDict()
        rv.update(self)
        return rv

    def _clean(self):
        if False:
            for i in range(10):
                print('nop')
        return list(self.items())

    def _compress(self, clean):
        if False:
            while True:
                i = 10
        return clean

    def _rollback(self, compressed):
        if False:
            for i in range(10):
                print('nop')
        self.clear()
        for (k, v) in compressed:
            self[k] = v

class RevertableSet(set):

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        if isinstance(state, tuple):
            self.update(state[0].keys())
        else:
            self.update(state)

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        rv = ({i: True for i in self},)
        return rv
    __reduce__ = object.__reduce__
    __reduce_ex__ = object.__reduce_ex__

    def __init__(self, *args):
        if False:
            print('Hello World!')
        log = renpy.game.log
        if log is not None:
            log.mutated[id(self)] = None
        set.__init__(self, *args)
    __iand__ = mutator(set.__iand__)
    __ior__ = mutator(set.__ior__)
    __isub__ = mutator(set.__isub__)
    __ixor__ = mutator(set.__ixor__)
    add = mutator(set.add)
    clear = mutator(set.clear)
    difference_update = mutator(set.difference_update)
    discard = mutator(set.discard)
    intersection_update = mutator(set.intersection_update)
    pop = mutator(set.pop)
    remove = mutator(set.remove)
    symmetric_difference_update = mutator(set.symmetric_difference_update)
    union_update = mutator(set.update)
    update = mutator(set.update)

    def wrapper(method):
        if False:
            i = 10
            return i + 15

        @_method_wrapper(method)
        def newmethod(*args, **kwargs):
            if False:
                return 10
            rv = method(*args, **kwargs)
            if isinstance(rv, set):
                return RevertableSet(rv)
            else:
                return rv
        return newmethod
    __and__ = wrapper(set.__and__)
    __sub__ = wrapper(set.__sub__)
    __xor__ = wrapper(set.__xor__)
    __or__ = wrapper(set.__or__)
    copy = wrapper(set.copy)
    difference = wrapper(set.difference)
    intersection = wrapper(set.intersection)
    symmetric_difference = wrapper(set.symmetric_difference)
    union = wrapper(set.union)
    del wrapper

    def _clean(self):
        if False:
            print('Hello World!')
        return list(self)

    def _compress(self, clean):
        if False:
            i = 10
            return i + 15
        return clean

    def _rollback(self, compressed):
        if False:
            for i in range(10):
                print('nop')
        set.clear(self)
        set.update(self, compressed)

class RevertableObject(object):

    def __new__(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self = super(RevertableObject, cls).__new__(cls)
        log = renpy.game.log
        if log is not None:
            log.mutated[id(self)] = None
        return self

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if (args or kwargs) and renpy.config.developer:
            raise TypeError('object() takes no parameters.')

    def __init_subclass__(cls):
        if False:
            while True:
                i = 10
        if renpy.config.developer and '__slots__' in cls.__dict__:
            raise TypeError('Classes with __slots__ do not support rollback. To create a class with slots, inherit from python_object instead.')
    __setattr__ = mutator(object.__setattr__)
    __delattr__ = mutator(object.__delattr__)

    def _clean(self):
        if False:
            return 10
        return self.__dict__.copy()

    def _compress(self, clean):
        if False:
            while True:
                i = 10
        return clean

    def _rollback(self, compressed):
        if False:
            print('Hello World!')
        self.__dict__.clear()
        self.__dict__.update(compressed)

def checkpointing(method):
    if False:
        for i in range(10):
            print('nop')

    @_method_wrapper(method)
    def do_checkpoint(self, *args, **kwargs):
        if False:
            return 10
        renpy.game.context().force_checkpoint = True
        return method(self, *args, **kwargs)
    return do_checkpoint

def list_wrapper(method):
    if False:
        i = 10
        return i + 15

    @_method_wrapper(method)
    def newmethod(*args, **kwargs):
        if False:
            print('Hello World!')
        l = method(*args, **kwargs)
        return RevertableList(l)
    return newmethod

class RollbackRandom(random.Random):
    """
    This is used for Random objects returned by renpy.random.Random.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        log = renpy.game.log
        if log is not None:
            log.mutated[id(self)] = None
        super(RollbackRandom, self).__init__()

    def _clean(self):
        if False:
            i = 10
            return i + 15
        return self.getstate()

    def _compress(self, clean):
        if False:
            i = 10
            return i + 15
        return clean

    def _rollback(self, compressed):
        if False:
            while True:
                i = 10
        super(RollbackRandom, self).setstate(compressed)
    setstate = checkpointing(mutator(random.Random.setstate))
    if PY2:
        jumpahead = checkpointing(mutator(random.Random.jumpahead))
    else:
        choices = list_wrapper(random.Random.choices)
    sample = list_wrapper(random.Random.sample)
    getrandbits = checkpointing(mutator(random.Random.getrandbits))
    seed = checkpointing(mutator(random.Random.seed))
    random = checkpointing(mutator(random.Random.random))

    def Random(self, seed=None):
        if False:
            return 10
        '\n        Returns a new RNG object separate from the main one.\n        '
        if seed is None:
            seed = self.random()
        new = RollbackRandom()
        new.seed(seed)
        return new

class DetRandom(random.Random):
    """
    This is renpy.random.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(DetRandom, self).__init__()
        self.stack = []
    if not PY2:
        choices = list_wrapper(random.Random.choices)
    sample = list_wrapper(random.Random.sample)

    def random(self):
        if False:
            while True:
                i = 10
        if self.stack:
            rv = self.stack.pop()
        else:
            rv = super(DetRandom, self).random()
        log = renpy.game.log
        if log.current is not None:
            log.current.random.append(rv)
        renpy.game.context().force_checkpoint = True
        return rv

    def pushback(self, l):
        if False:
            return 10
        '\n        Pushes the random numbers in l onto the stack so they will be generated\n        in the order given.\n        '
        ll = l[:]
        ll.reverse()
        self.stack.extend(ll)

    def reset(self):
        if False:
            while True:
                i = 10
        '\n        Resets the RNG, removing all of the pushbacked numbers.\n        '
        del self.stack[:]

    def Random(self, seed=None):
        if False:
            print('Hello World!')
        '\n        Returns a new RNG object separate from the main one.\n        '
        if seed is None:
            seed = self.random()
        new = RollbackRandom()
        new.seed(seed)
        return new