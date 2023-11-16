from abc import abstractmethod, ABCMeta
from collections.abc import Sequence, Hashable
from numbers import Integral
import operator
from pyrsistent._transformations import transform

def _bitcount(val):
    if False:
        for i in range(10):
            print('nop')
    return bin(val).count('1')
BRANCH_FACTOR = 32
BIT_MASK = BRANCH_FACTOR - 1
SHIFT = _bitcount(BIT_MASK)

def compare_pvector(v, other, operator):
    if False:
        for i in range(10):
            print('nop')
    return operator(v.tolist(), other.tolist() if isinstance(other, PVector) else other)

def _index_or_slice(index, stop):
    if False:
        i = 10
        return i + 15
    if stop is None:
        return index
    return slice(index, stop)

class PythonPVector(object):
    """
    Support structure for PVector that implements structural sharing for vectors using a trie.
    """
    __slots__ = ('_count', '_shift', '_root', '_tail', '_tail_offset', '__weakref__')

    def __new__(cls, count, shift, root, tail):
        if False:
            print('Hello World!')
        self = super(PythonPVector, cls).__new__(cls)
        self._count = count
        self._shift = shift
        self._root = root
        self._tail = tail
        self._tail_offset = self._count - len(self._tail)
        return self

    def __len__(self):
        if False:
            while True:
                i = 10
        return self._count

    def __getitem__(self, index):
        if False:
            print('Hello World!')
        if isinstance(index, slice):
            if index.start is None and index.stop is None and (index.step is None):
                return self
            return _EMPTY_PVECTOR.extend(self.tolist()[index])
        if index < 0:
            index += self._count
        return PythonPVector._node_for(self, index)[index & BIT_MASK]

    def __add__(self, other):
        if False:
            print('Hello World!')
        return self.extend(other)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'pvector({0})'.format(str(self.tolist()))

    def __str__(self):
        if False:
            return 10
        return self.__repr__()

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self.tolist())

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        return not self.__eq__(other)

    def __eq__(self, other):
        if False:
            return 10
        return self is other or ((hasattr(other, '__len__') and self._count == len(other)) and compare_pvector(self, other, operator.eq))

    def __gt__(self, other):
        if False:
            while True:
                i = 10
        return compare_pvector(self, other, operator.gt)

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        return compare_pvector(self, other, operator.lt)

    def __ge__(self, other):
        if False:
            return 10
        return compare_pvector(self, other, operator.ge)

    def __le__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return compare_pvector(self, other, operator.le)

    def __mul__(self, times):
        if False:
            print('Hello World!')
        if times <= 0 or self is _EMPTY_PVECTOR:
            return _EMPTY_PVECTOR
        if times == 1:
            return self
        return _EMPTY_PVECTOR.extend(times * self.tolist())
    __rmul__ = __mul__

    def _fill_list(self, node, shift, the_list):
        if False:
            i = 10
            return i + 15
        if shift:
            shift -= SHIFT
            for n in node:
                self._fill_list(n, shift, the_list)
        else:
            the_list.extend(node)

    def tolist(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The fastest way to convert the vector into a python list.\n        '
        the_list = []
        self._fill_list(self._root, self._shift, the_list)
        the_list.extend(self._tail)
        return the_list

    def _totuple(self):
        if False:
            print('Hello World!')
        '\n        Returns the content as a python tuple.\n        '
        return tuple(self.tolist())

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(self._totuple())

    def transform(self, *transformations):
        if False:
            for i in range(10):
                print('nop')
        return transform(self, transformations)

    def __reduce__(self):
        if False:
            print('Hello World!')
        return (pvector, (self.tolist(),))

    def mset(self, *args):
        if False:
            return 10
        if len(args) % 2:
            raise TypeError('mset expected an even number of arguments')
        evolver = self.evolver()
        for i in range(0, len(args), 2):
            evolver[args[i]] = args[i + 1]
        return evolver.persistent()

    class Evolver(object):
        __slots__ = ('_count', '_shift', '_root', '_tail', '_tail_offset', '_dirty_nodes', '_extra_tail', '_cached_leafs', '_orig_pvector')

        def __init__(self, v):
            if False:
                for i in range(10):
                    print('nop')
            self._reset(v)

        def __getitem__(self, index):
            if False:
                print('Hello World!')
            if not isinstance(index, Integral):
                raise TypeError("'%s' object cannot be interpreted as an index" % type(index).__name__)
            if index < 0:
                index += self._count + len(self._extra_tail)
            if self._count <= index < self._count + len(self._extra_tail):
                return self._extra_tail[index - self._count]
            return PythonPVector._node_for(self, index)[index & BIT_MASK]

        def _reset(self, v):
            if False:
                print('Hello World!')
            self._count = v._count
            self._shift = v._shift
            self._root = v._root
            self._tail = v._tail
            self._tail_offset = v._tail_offset
            self._dirty_nodes = {}
            self._cached_leafs = {}
            self._extra_tail = []
            self._orig_pvector = v

        def append(self, element):
            if False:
                while True:
                    i = 10
            self._extra_tail.append(element)
            return self

        def extend(self, iterable):
            if False:
                i = 10
                return i + 15
            self._extra_tail.extend(iterable)
            return self

        def set(self, index, val):
            if False:
                for i in range(10):
                    print('nop')
            self[index] = val
            return self

        def __setitem__(self, index, val):
            if False:
                i = 10
                return i + 15
            if not isinstance(index, Integral):
                raise TypeError("'%s' object cannot be interpreted as an index" % type(index).__name__)
            if index < 0:
                index += self._count + len(self._extra_tail)
            if 0 <= index < self._count:
                node = self._cached_leafs.get(index >> SHIFT)
                if node:
                    node[index & BIT_MASK] = val
                elif index >= self._tail_offset:
                    if id(self._tail) not in self._dirty_nodes:
                        self._tail = list(self._tail)
                        self._dirty_nodes[id(self._tail)] = True
                        self._cached_leafs[index >> SHIFT] = self._tail
                    self._tail[index & BIT_MASK] = val
                else:
                    self._root = self._do_set(self._shift, self._root, index, val)
            elif self._count <= index < self._count + len(self._extra_tail):
                self._extra_tail[index - self._count] = val
            elif index == self._count + len(self._extra_tail):
                self._extra_tail.append(val)
            else:
                raise IndexError('Index out of range: %s' % (index,))

        def _do_set(self, level, node, i, val):
            if False:
                for i in range(10):
                    print('nop')
            if id(node) in self._dirty_nodes:
                ret = node
            else:
                ret = list(node)
                self._dirty_nodes[id(ret)] = True
            if level == 0:
                ret[i & BIT_MASK] = val
                self._cached_leafs[i >> SHIFT] = ret
            else:
                sub_index = i >> level & BIT_MASK
                ret[sub_index] = self._do_set(level - SHIFT, node[sub_index], i, val)
            return ret

        def delete(self, index):
            if False:
                return 10
            del self[index]
            return self

        def __delitem__(self, key):
            if False:
                while True:
                    i = 10
            if self._orig_pvector:
                l = PythonPVector(self._count, self._shift, self._root, self._tail).tolist()
                l.extend(self._extra_tail)
                self._reset(_EMPTY_PVECTOR)
                self._extra_tail = l
            del self._extra_tail[key]

        def persistent(self):
            if False:
                while True:
                    i = 10
            result = self._orig_pvector
            if self.is_dirty():
                result = PythonPVector(self._count, self._shift, self._root, self._tail).extend(self._extra_tail)
                self._reset(result)
            return result

        def __len__(self):
            if False:
                for i in range(10):
                    print('nop')
            return self._count + len(self._extra_tail)

        def is_dirty(self):
            if False:
                print('Hello World!')
            return bool(self._dirty_nodes or self._extra_tail)

    def evolver(self):
        if False:
            return 10
        return PythonPVector.Evolver(self)

    def set(self, i, val):
        if False:
            i = 10
            return i + 15
        if not isinstance(i, Integral):
            raise TypeError("'%s' object cannot be interpreted as an index" % type(i).__name__)
        if i < 0:
            i += self._count
        if 0 <= i < self._count:
            if i >= self._tail_offset:
                new_tail = list(self._tail)
                new_tail[i & BIT_MASK] = val
                return PythonPVector(self._count, self._shift, self._root, new_tail)
            return PythonPVector(self._count, self._shift, self._do_set(self._shift, self._root, i, val), self._tail)
        if i == self._count:
            return self.append(val)
        raise IndexError('Index out of range: %s' % (i,))

    def _do_set(self, level, node, i, val):
        if False:
            while True:
                i = 10
        ret = list(node)
        if level == 0:
            ret[i & BIT_MASK] = val
        else:
            sub_index = i >> level & BIT_MASK
            ret[sub_index] = self._do_set(level - SHIFT, node[sub_index], i, val)
        return ret

    @staticmethod
    def _node_for(pvector_like, i):
        if False:
            i = 10
            return i + 15
        if 0 <= i < pvector_like._count:
            if i >= pvector_like._tail_offset:
                return pvector_like._tail
            node = pvector_like._root
            for level in range(pvector_like._shift, 0, -SHIFT):
                node = node[i >> level & BIT_MASK]
            return node
        raise IndexError('Index out of range: %s' % (i,))

    def _create_new_root(self):
        if False:
            i = 10
            return i + 15
        new_shift = self._shift
        if self._count >> SHIFT > 1 << self._shift:
            new_root = [self._root, self._new_path(self._shift, self._tail)]
            new_shift += SHIFT
        else:
            new_root = self._push_tail(self._shift, self._root, self._tail)
        return (new_root, new_shift)

    def append(self, val):
        if False:
            return 10
        if len(self._tail) < BRANCH_FACTOR:
            new_tail = list(self._tail)
            new_tail.append(val)
            return PythonPVector(self._count + 1, self._shift, self._root, new_tail)
        (new_root, new_shift) = self._create_new_root()
        return PythonPVector(self._count + 1, new_shift, new_root, [val])

    def _new_path(self, level, node):
        if False:
            return 10
        if level == 0:
            return node
        return [self._new_path(level - SHIFT, node)]

    def _mutating_insert_tail(self):
        if False:
            while True:
                i = 10
        (self._root, self._shift) = self._create_new_root()
        self._tail = []

    def _mutating_fill_tail(self, offset, sequence):
        if False:
            print('Hello World!')
        max_delta_len = BRANCH_FACTOR - len(self._tail)
        delta = sequence[offset:offset + max_delta_len]
        self._tail.extend(delta)
        delta_len = len(delta)
        self._count += delta_len
        return offset + delta_len

    def _mutating_extend(self, sequence):
        if False:
            while True:
                i = 10
        offset = 0
        sequence_len = len(sequence)
        while offset < sequence_len:
            offset = self._mutating_fill_tail(offset, sequence)
            if len(self._tail) == BRANCH_FACTOR:
                self._mutating_insert_tail()
        self._tail_offset = self._count - len(self._tail)

    def extend(self, obj):
        if False:
            while True:
                i = 10
        l = obj.tolist() if isinstance(obj, PythonPVector) else list(obj)
        if l:
            new_vector = self.append(l[0])
            new_vector._mutating_extend(l[1:])
            return new_vector
        return self

    def _push_tail(self, level, parent, tail_node):
        if False:
            i = 10
            return i + 15
        '\n        if parent is leaf, insert node,\n        else does it map to an existing child? ->\n             node_to_insert = push node one more level\n        else alloc new path\n\n        return  node_to_insert placed in copy of parent\n        '
        ret = list(parent)
        if level == SHIFT:
            ret.append(tail_node)
            return ret
        sub_index = self._count - 1 >> level & BIT_MASK
        if len(parent) > sub_index:
            ret[sub_index] = self._push_tail(level - SHIFT, parent[sub_index], tail_node)
            return ret
        ret.append(self._new_path(level - SHIFT, tail_node))
        return ret

    def index(self, value, *args, **kwargs):
        if False:
            print('Hello World!')
        return self.tolist().index(value, *args, **kwargs)

    def count(self, value):
        if False:
            for i in range(10):
                print('nop')
        return self.tolist().count(value)

    def delete(self, index, stop=None):
        if False:
            print('Hello World!')
        l = self.tolist()
        del l[_index_or_slice(index, stop)]
        return _EMPTY_PVECTOR.extend(l)

    def remove(self, value):
        if False:
            while True:
                i = 10
        l = self.tolist()
        l.remove(value)
        return _EMPTY_PVECTOR.extend(l)

class PVector(metaclass=ABCMeta):
    """
    Persistent vector implementation. Meant as a replacement for the cases where you would normally
    use a Python list.

    Do not instantiate directly, instead use the factory functions :py:func:`v` and :py:func:`pvector` to
    create an instance.

    Heavily influenced by the persistent vector available in Clojure. Initially this was more or
    less just a port of the Java code for the Clojure vector. It has since been modified and to
    some extent optimized for usage in Python.

    The vector is organized as a trie, any mutating method will return a new vector that contains the changes. No
    updates are done to the original vector. Structural sharing between vectors are applied where possible to save
    space and to avoid making complete copies.

    This structure corresponds most closely to the built in list type and is intended as a replacement. Where the
    semantics are the same (more or less) the same function names have been used but for some cases it is not possible,
    for example assignments.

    The PVector implements the Sequence protocol and is Hashable.

    Inserts are amortized O(1). Random access is log32(n) where n is the size of the vector.

    The following are examples of some common operations on persistent vectors:

    >>> p = v(1, 2, 3)
    >>> p2 = p.append(4)
    >>> p3 = p2.extend([5, 6, 7])
    >>> p
    pvector([1, 2, 3])
    >>> p2
    pvector([1, 2, 3, 4])
    >>> p3
    pvector([1, 2, 3, 4, 5, 6, 7])
    >>> p3[5]
    6
    >>> p.set(1, 99)
    pvector([1, 99, 3])
    >>>
    """

    @abstractmethod
    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        >>> len(v(1, 2, 3))\n        3\n        '

    @abstractmethod
    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get value at index. Full slicing support.\n\n        >>> v1 = v(5, 6, 7, 8)\n        >>> v1[2]\n        7\n        >>> v1[1:3]\n        pvector([6, 7])\n        '

    @abstractmethod
    def __add__(self, other):
        if False:
            print('Hello World!')
        '\n        >>> v1 = v(1, 2)\n        >>> v2 = v(3, 4)\n        >>> v1 + v2\n        pvector([1, 2, 3, 4])\n        '

    @abstractmethod
    def __mul__(self, times):
        if False:
            i = 10
            return i + 15
        '\n        >>> v1 = v(1, 2)\n        >>> 3 * v1\n        pvector([1, 2, 1, 2, 1, 2])\n        '

    @abstractmethod
    def __hash__(self):
        if False:
            return 10
        '\n        >>> v1 = v(1, 2, 3)\n        >>> v2 = v(1, 2, 3)\n        >>> hash(v1) == hash(v2)\n        True\n        '

    @abstractmethod
    def evolver(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new evolver for this pvector. The evolver acts as a mutable view of the vector\n        with "transaction like" semantics. No part of the underlying vector i updated, it is still\n        fully immutable. Furthermore multiple evolvers created from the same pvector do not\n        interfere with each other.\n\n        You may want to use an evolver instead of working directly with the pvector in the\n        following cases:\n\n        * Multiple updates are done to the same vector and the intermediate results are of no\n          interest. In this case using an evolver may be a more efficient and easier to work with.\n        * You need to pass a vector into a legacy function or a function that you have no control\n          over which performs in place mutations of lists. In this case pass an evolver instance\n          instead and then create a new pvector from the evolver once the function returns.\n\n        The following example illustrates a typical workflow when working with evolvers. It also\n        displays most of the API (which i kept small by design, you should not be tempted to\n        use evolvers in excess ;-)).\n\n        Create the evolver and perform various mutating updates to it:\n\n        >>> v1 = v(1, 2, 3, 4, 5)\n        >>> e = v1.evolver()\n        >>> e[1] = 22\n        >>> _ = e.append(6)\n        >>> _ = e.extend([7, 8, 9])\n        >>> e[8] += 1\n        >>> len(e)\n        9\n\n        The underlying pvector remains the same:\n\n        >>> v1\n        pvector([1, 2, 3, 4, 5])\n\n        The changes are kept in the evolver. An updated pvector can be created using the\n        persistent() function on the evolver.\n\n        >>> v2 = e.persistent()\n        >>> v2\n        pvector([1, 22, 3, 4, 5, 6, 7, 8, 10])\n\n        The new pvector will share data with the original pvector in the same way that would have\n        been done if only using operations on the pvector.\n        '

    @abstractmethod
    def mset(self, *args):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a new vector with elements in specified positions replaced by values (multi set).\n\n        Elements on even positions in the argument list are interpreted as indexes while\n        elements on odd positions are considered values.\n\n        >>> v1 = v(1, 2, 3)\n        >>> v1.mset(0, 11, 2, 33)\n        pvector([11, 2, 33])\n        '

    @abstractmethod
    def set(self, i, val):
        if False:
            i = 10
            return i + 15
        '\n        Return a new vector with element at position i replaced with val. The original vector remains unchanged.\n\n        Setting a value one step beyond the end of the vector is equal to appending. Setting beyond that will\n        result in an IndexError.\n\n        >>> v1 = v(1, 2, 3)\n        >>> v1.set(1, 4)\n        pvector([1, 4, 3])\n        >>> v1.set(3, 4)\n        pvector([1, 2, 3, 4])\n        >>> v1.set(-1, 4)\n        pvector([1, 2, 4])\n        '

    @abstractmethod
    def append(self, val):
        if False:
            print('Hello World!')
        '\n        Return a new vector with val appended.\n\n        >>> v1 = v(1, 2)\n        >>> v1.append(3)\n        pvector([1, 2, 3])\n        '

    @abstractmethod
    def extend(self, obj):
        if False:
            while True:
                i = 10
        '\n        Return a new vector with all values in obj appended to it. Obj may be another\n        PVector or any other Iterable.\n\n        >>> v1 = v(1, 2, 3)\n        >>> v1.extend([4, 5])\n        pvector([1, 2, 3, 4, 5])\n        '

    @abstractmethod
    def index(self, value, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Return first index of value. Additional indexes may be supplied to limit the search to a\n        sub range of the vector.\n\n        >>> v1 = v(1, 2, 3, 4, 3)\n        >>> v1.index(3)\n        2\n        >>> v1.index(3, 3, 5)\n        4\n        '

    @abstractmethod
    def count(self, value):
        if False:
            while True:
                i = 10
        '\n        Return the number of times that value appears in the vector.\n\n        >>> v1 = v(1, 4, 3, 4)\n        >>> v1.count(4)\n        2\n        '

    @abstractmethod
    def transform(self, *transformations):
        if False:
            return 10
        "\n        Transform arbitrarily complex combinations of PVectors and PMaps. A transformation\n        consists of two parts. One match expression that specifies which elements to transform\n        and one transformation function that performs the actual transformation.\n\n        >>> from pyrsistent import freeze, ny\n        >>> news_paper = freeze({'articles': [{'author': 'Sara', 'content': 'A short article'},\n        ...                                   {'author': 'Steve', 'content': 'A slightly longer article'}],\n        ...                      'weather': {'temperature': '11C', 'wind': '5m/s'}})\n        >>> short_news = news_paper.transform(['articles', ny, 'content'], lambda c: c[:25] + '...' if len(c) > 25 else c)\n        >>> very_short_news = news_paper.transform(['articles', ny, 'content'], lambda c: c[:15] + '...' if len(c) > 15 else c)\n        >>> very_short_news.articles[0].content\n        'A short article'\n        >>> very_short_news.articles[1].content\n        'A slightly long...'\n\n        When nothing has been transformed the original data structure is kept\n\n        >>> short_news is news_paper\n        True\n        >>> very_short_news is news_paper\n        False\n        >>> very_short_news.articles[0] is news_paper.articles[0]\n        True\n        "

    @abstractmethod
    def delete(self, index, stop=None):
        if False:
            print('Hello World!')
        '\n        Delete a portion of the vector by index or range.\n\n        >>> v1 = v(1, 2, 3, 4, 5)\n        >>> v1.delete(1)\n        pvector([1, 3, 4, 5])\n        >>> v1.delete(1, 3)\n        pvector([1, 4, 5])\n        '

    @abstractmethod
    def remove(self, value):
        if False:
            i = 10
            return i + 15
        '\n        Remove the first occurrence of a value from the vector.\n\n        >>> v1 = v(1, 2, 3, 2, 1)\n        >>> v2 = v1.remove(1)\n        >>> v2\n        pvector([2, 3, 2, 1])\n        >>> v2.remove(1)\n        pvector([2, 3, 2])\n        '
_EMPTY_PVECTOR = PythonPVector(0, SHIFT, [], [])
PVector.register(PythonPVector)
Sequence.register(PVector)
Hashable.register(PVector)

def python_pvector(iterable=()):
    if False:
        i = 10
        return i + 15
    '\n    Create a new persistent vector containing the elements in iterable.\n\n    >>> v1 = pvector([1, 2, 3])\n    >>> v1\n    pvector([1, 2, 3])\n    '
    return _EMPTY_PVECTOR.extend(iterable)
try:
    import os
    if os.environ.get('PYRSISTENT_NO_C_EXTENSION'):
        pvector = python_pvector
    else:
        from pvectorc import pvector
        PVector.register(type(pvector()))
except ImportError:
    pvector = python_pvector

def v(*elements):
    if False:
        print('Hello World!')
    '\n    Create a new persistent vector containing all parameters to this function.\n\n    >>> v1 = v(1, 2, 3)\n    >>> v1\n    pvector([1, 2, 3])\n    '
    return pvector(elements)