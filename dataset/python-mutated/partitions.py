from sympy.core import Basic, Dict, sympify, Tuple
from sympy.core.numbers import Integer
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import _sympify
from sympy.functions.combinatorial.numbers import bell
from sympy.matrices import zeros
from sympy.sets.sets import FiniteSet, Union
from sympy.utilities.iterables import flatten, group
from sympy.utilities.misc import as_int
from collections import defaultdict

class Partition(FiniteSet):
    """
    This class represents an abstract partition.

    A partition is a set of disjoint sets whose union equals a given set.

    See Also
    ========

    sympy.utilities.iterables.partitions,
    sympy.utilities.iterables.multiset_partitions
    """
    _rank = None
    _partition = None

    def __new__(cls, *partition):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generates a new partition object.\n\n        This method also verifies if the arguments passed are\n        valid and raises a ValueError if they are not.\n\n        Examples\n        ========\n\n        Creating Partition from Python lists:\n\n        >>> from sympy.combinatorics import Partition\n        >>> a = Partition([1, 2], [3])\n        >>> a\n        Partition({3}, {1, 2})\n        >>> a.partition\n        [[1, 2], [3]]\n        >>> len(a)\n        2\n        >>> a.members\n        (1, 2, 3)\n\n        Creating Partition from Python sets:\n\n        >>> Partition({1, 2, 3}, {4, 5})\n        Partition({4, 5}, {1, 2, 3})\n\n        Creating Partition from SymPy finite sets:\n\n        >>> from sympy import FiniteSet\n        >>> a = FiniteSet(1, 2, 3)\n        >>> b = FiniteSet(4, 5)\n        >>> Partition(a, b)\n        Partition({4, 5}, {1, 2, 3})\n        '
        args = []
        dups = False
        for arg in partition:
            if isinstance(arg, list):
                as_set = set(arg)
                if len(as_set) < len(arg):
                    dups = True
                    break
                arg = as_set
            args.append(_sympify(arg))
        if not all((isinstance(part, FiniteSet) for part in args)):
            raise ValueError('Each argument to Partition should be a list, set, or a FiniteSet')
        U = Union(*args)
        if dups or len(U) < sum((len(arg) for arg in args)):
            raise ValueError('Partition contained duplicate elements.')
        obj = FiniteSet.__new__(cls, *args)
        obj.members = tuple(U)
        obj.size = len(U)
        return obj

    def sort_key(self, order=None):
        if False:
            for i in range(10):
                print('nop')
        'Return a canonical key that can be used for sorting.\n\n        Ordering is based on the size and sorted elements of the partition\n        and ties are broken with the rank.\n\n        Examples\n        ========\n\n        >>> from sympy import default_sort_key\n        >>> from sympy.combinatorics import Partition\n        >>> from sympy.abc import x\n        >>> a = Partition([1, 2])\n        >>> b = Partition([3, 4])\n        >>> c = Partition([1, x])\n        >>> d = Partition(list(range(4)))\n        >>> l = [d, b, a + 1, a, c]\n        >>> l.sort(key=default_sort_key); l\n        [Partition({1, 2}), Partition({1}, {2}), Partition({1, x}), Partition({3, 4}), Partition({0, 1, 2, 3})]\n        '
        if order is None:
            members = self.members
        else:
            members = tuple(sorted(self.members, key=lambda w: default_sort_key(w, order)))
        return tuple(map(default_sort_key, (self.size, members, self.rank)))

    @property
    def partition(self):
        if False:
            return 10
        'Return partition as a sorted list of lists.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Partition\n        >>> Partition([1], [2, 3]).partition\n        [[1], [2, 3]]\n        '
        if self._partition is None:
            self._partition = sorted([sorted(p, key=default_sort_key) for p in self.args])
        return self._partition

    def __add__(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Return permutation whose rank is ``other`` greater than current rank,\n        (mod the maximum rank for the set).\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Partition\n        >>> a = Partition([1, 2], [3])\n        >>> a.rank\n        1\n        >>> (a + 1).rank\n        2\n        >>> (a + 100).rank\n        1\n        '
        other = as_int(other)
        offset = self.rank + other
        result = RGS_unrank(offset % RGS_enum(self.size), self.size)
        return Partition.from_rgs(result, self.members)

    def __sub__(self, other):
        if False:
            while True:
                i = 10
        '\n        Return permutation whose rank is ``other`` less than current rank,\n        (mod the maximum rank for the set).\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Partition\n        >>> a = Partition([1, 2], [3])\n        >>> a.rank\n        1\n        >>> (a - 1).rank\n        0\n        >>> (a - 100).rank\n        1\n        '
        return self.__add__(-other)

    def __le__(self, other):
        if False:
            return 10
        '\n        Checks if a partition is less than or equal to\n        the other based on rank.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Partition\n        >>> a = Partition([1, 2], [3, 4, 5])\n        >>> b = Partition([1], [2, 3], [4], [5])\n        >>> a.rank, b.rank\n        (9, 34)\n        >>> a <= a\n        True\n        >>> a <= b\n        True\n        '
        return self.sort_key() <= sympify(other).sort_key()

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks if a partition is less than the other.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Partition\n        >>> a = Partition([1, 2], [3, 4, 5])\n        >>> b = Partition([1], [2, 3], [4], [5])\n        >>> a.rank, b.rank\n        (9, 34)\n        >>> a < b\n        True\n        '
        return self.sort_key() < sympify(other).sort_key()

    @property
    def rank(self):
        if False:
            return 10
        '\n        Gets the rank of a partition.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Partition\n        >>> a = Partition([1, 2], [3], [4, 5])\n        >>> a.rank\n        13\n        '
        if self._rank is not None:
            return self._rank
        self._rank = RGS_rank(self.RGS)
        return self._rank

    @property
    def RGS(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the "restricted growth string" of the partition.\n\n        Explanation\n        ===========\n\n        The RGS is returned as a list of indices, L, where L[i] indicates\n        the block in which element i appears. For example, in a partition\n        of 3 elements (a, b, c) into 2 blocks ([c], [a, b]) the RGS is\n        [1, 1, 0]: "a" is in block 1, "b" is in block 1 and "c" is in block 0.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Partition\n        >>> a = Partition([1, 2], [3], [4, 5])\n        >>> a.members\n        (1, 2, 3, 4, 5)\n        >>> a.RGS\n        (0, 0, 1, 2, 2)\n        >>> a + 1\n        Partition({3}, {4}, {5}, {1, 2})\n        >>> _.RGS\n        (0, 0, 1, 2, 3)\n        '
        rgs = {}
        partition = self.partition
        for (i, part) in enumerate(partition):
            for j in part:
                rgs[j] = i
        return tuple([rgs[i] for i in sorted([i for p in partition for i in p], key=default_sort_key)])

    @classmethod
    def from_rgs(self, rgs, elements):
        if False:
            while True:
                i = 10
        "\n        Creates a set partition from a restricted growth string.\n\n        Explanation\n        ===========\n\n        The indices given in rgs are assumed to be the index\n        of the element as given in elements *as provided* (the\n        elements are not sorted by this routine). Block numbering\n        starts from 0. If any block was not referenced in ``rgs``\n        an error will be raised.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Partition\n        >>> Partition.from_rgs([0, 1, 2, 0, 1], list('abcde'))\n        Partition({c}, {a, d}, {b, e})\n        >>> Partition.from_rgs([0, 1, 2, 0, 1], list('cbead'))\n        Partition({e}, {a, c}, {b, d})\n        >>> a = Partition([1, 4], [2], [3, 5])\n        >>> Partition.from_rgs(a.RGS, a.members)\n        Partition({2}, {1, 4}, {3, 5})\n        "
        if len(rgs) != len(elements):
            raise ValueError('mismatch in rgs and element lengths')
        max_elem = max(rgs) + 1
        partition = [[] for i in range(max_elem)]
        j = 0
        for i in rgs:
            partition[i].append(elements[j])
            j += 1
        if not all((p for p in partition)):
            raise ValueError('some blocks of the partition were empty.')
        return Partition(*partition)

class IntegerPartition(Basic):
    """
    This class represents an integer partition.

    Explanation
    ===========

    In number theory and combinatorics, a partition of a positive integer,
    ``n``, also called an integer partition, is a way of writing ``n`` as a
    list of positive integers that sum to n. Two partitions that differ only
    in the order of summands are considered to be the same partition; if order
    matters then the partitions are referred to as compositions. For example,
    4 has five partitions: [4], [3, 1], [2, 2], [2, 1, 1], and [1, 1, 1, 1];
    the compositions [1, 2, 1] and [1, 1, 2] are the same as partition
    [2, 1, 1].

    See Also
    ========

    sympy.utilities.iterables.partitions,
    sympy.utilities.iterables.multiset_partitions

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Partition_%28number_theory%29
    """
    _dict = None
    _keys = None

    def __new__(cls, partition, integer=None):
        if False:
            print('Hello World!')
        '\n        Generates a new IntegerPartition object from a list or dictionary.\n\n        Explanation\n        ===========\n\n        The partition can be given as a list of positive integers or a\n        dictionary of (integer, multiplicity) items. If the partition is\n        preceded by an integer an error will be raised if the partition\n        does not sum to that given integer.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.partitions import IntegerPartition\n        >>> a = IntegerPartition([5, 4, 3, 1, 1])\n        >>> a\n        IntegerPartition(14, (5, 4, 3, 1, 1))\n        >>> print(a)\n        [5, 4, 3, 1, 1]\n        >>> IntegerPartition({1:3, 2:1})\n        IntegerPartition(5, (2, 1, 1, 1))\n\n        If the value that the partition should sum to is given first, a check\n        will be made to see n error will be raised if there is a discrepancy:\n\n        >>> IntegerPartition(10, [5, 4, 3, 1])\n        Traceback (most recent call last):\n        ...\n        ValueError: The partition is not valid\n\n        '
        if integer is not None:
            (integer, partition) = (partition, integer)
        if isinstance(partition, (dict, Dict)):
            _ = []
            for (k, v) in sorted(partition.items(), reverse=True):
                if not v:
                    continue
                (k, v) = (as_int(k), as_int(v))
                _.extend([k] * v)
            partition = tuple(_)
        else:
            partition = tuple(sorted(map(as_int, partition), reverse=True))
        sum_ok = False
        if integer is None:
            integer = sum(partition)
            sum_ok = True
        else:
            integer = as_int(integer)
        if not sum_ok and sum(partition) != integer:
            raise ValueError('Partition did not add to %s' % integer)
        if any((i < 1 for i in partition)):
            raise ValueError('All integer summands must be greater than one')
        obj = Basic.__new__(cls, Integer(integer), Tuple(*partition))
        obj.partition = list(partition)
        obj.integer = integer
        return obj

    def prev_lex(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the previous partition of the integer, n, in lexical order,\n        wrapping around to [1, ..., 1] if the partition is [n].\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.partitions import IntegerPartition\n        >>> p = IntegerPartition([4])\n        >>> print(p.prev_lex())\n        [3, 1]\n        >>> p.partition > p.prev_lex().partition\n        True\n        '
        d = defaultdict(int)
        d.update(self.as_dict())
        keys = self._keys
        if keys == [1]:
            return IntegerPartition({self.integer: 1})
        if keys[-1] != 1:
            d[keys[-1]] -= 1
            if keys[-1] == 2:
                d[1] = 2
            else:
                d[keys[-1] - 1] = d[1] = 1
        else:
            d[keys[-2]] -= 1
            left = d[1] + keys[-2]
            new = keys[-2]
            d[1] = 0
            while left:
                new -= 1
                if left - new >= 0:
                    d[new] += left // new
                    left -= d[new] * new
        return IntegerPartition(self.integer, d)

    def next_lex(self):
        if False:
            print('Hello World!')
        'Return the next partition of the integer, n, in lexical order,\n        wrapping around to [n] if the partition is [1, ..., 1].\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.partitions import IntegerPartition\n        >>> p = IntegerPartition([3, 1])\n        >>> print(p.next_lex())\n        [4]\n        >>> p.partition < p.next_lex().partition\n        True\n        '
        d = defaultdict(int)
        d.update(self.as_dict())
        key = self._keys
        a = key[-1]
        if a == self.integer:
            d.clear()
            d[1] = self.integer
        elif a == 1:
            if d[a] > 1:
                d[a + 1] += 1
                d[a] -= 2
            else:
                b = key[-2]
                d[b + 1] += 1
                d[1] = (d[b] - 1) * b
                d[b] = 0
        elif d[a] > 1:
            if len(key) == 1:
                d.clear()
                d[a + 1] = 1
                d[1] = self.integer - a - 1
            else:
                a1 = a + 1
                d[a1] += 1
                d[1] = d[a] * a - a1
                d[a] = 0
        else:
            b = key[-2]
            b1 = b + 1
            d[b1] += 1
            need = d[b] * b + d[a] * a - b1
            d[a] = d[b] = 0
            d[1] = need
        return IntegerPartition(self.integer, d)

    def as_dict(self):
        if False:
            i = 10
            return i + 15
        'Return the partition as a dictionary whose keys are the\n        partition integers and the values are the multiplicity of that\n        integer.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.partitions import IntegerPartition\n        >>> IntegerPartition([1]*3 + [2] + [3]*4).as_dict()\n        {1: 3, 2: 1, 3: 4}\n        '
        if self._dict is None:
            groups = group(self.partition, multiple=False)
            self._keys = [g[0] for g in groups]
            self._dict = dict(groups)
        return self._dict

    @property
    def conjugate(self):
        if False:
            return 10
        '\n        Computes the conjugate partition of itself.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.partitions import IntegerPartition\n        >>> a = IntegerPartition([6, 3, 3, 2, 1])\n        >>> a.conjugate\n        [5, 4, 3, 1, 1, 1]\n        '
        j = 1
        temp_arr = list(self.partition) + [0]
        k = temp_arr[0]
        b = [0] * k
        while k > 0:
            while k > temp_arr[j]:
                b[k - 1] = j
                k -= 1
            j += 1
        return b

    def __lt__(self, other):
        if False:
            print('Hello World!')
        'Return True if self is less than other when the partition\n        is listed from smallest to biggest.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.partitions import IntegerPartition\n        >>> a = IntegerPartition([3, 1])\n        >>> a < a\n        False\n        >>> b = a.next_lex()\n        >>> a < b\n        True\n        >>> a == b\n        False\n        '
        return list(reversed(self.partition)) < list(reversed(other.partition))

    def __le__(self, other):
        if False:
            i = 10
            return i + 15
        'Return True if self is less than other when the partition\n        is listed from smallest to biggest.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.partitions import IntegerPartition\n        >>> a = IntegerPartition([4])\n        >>> a <= a\n        True\n        '
        return list(reversed(self.partition)) <= list(reversed(other.partition))

    def as_ferrers(self, char='#'):
        if False:
            while True:
                i = 10
        '\n        Prints the ferrer diagram of a partition.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.partitions import IntegerPartition\n        >>> print(IntegerPartition([1, 1, 5]).as_ferrers())\n        #####\n        #\n        #\n        '
        return '\n'.join([char * i for i in self.partition])

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(list(self.partition))

def random_integer_partition(n, seed=None):
    if False:
        while True:
            i = 10
    '\n    Generates a random integer partition summing to ``n`` as a list\n    of reverse-sorted integers.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.partitions import random_integer_partition\n\n    For the following, a seed is given so a known value can be shown; in\n    practice, the seed would not be given.\n\n    >>> random_integer_partition(100, seed=[1, 1, 12, 1, 2, 1, 85, 1])\n    [85, 12, 2, 1]\n    >>> random_integer_partition(10, seed=[1, 2, 3, 1, 5, 1])\n    [5, 3, 1, 1]\n    >>> random_integer_partition(1)\n    [1]\n    '
    from sympy.core.random import _randint
    n = as_int(n)
    if n < 1:
        raise ValueError('n must be a positive integer')
    randint = _randint(seed)
    partition = []
    while n > 0:
        k = randint(1, n)
        mult = randint(1, n // k)
        partition.append((k, mult))
        n -= k * mult
    partition.sort(reverse=True)
    partition = flatten([[k] * m for (k, m) in partition])
    return partition

def RGS_generalized(m):
    if False:
        i = 10
        return i + 15
    '\n    Computes the m + 1 generalized unrestricted growth strings\n    and returns them as rows in matrix.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.partitions import RGS_generalized\n    >>> RGS_generalized(6)\n    Matrix([\n    [  1,   1,   1,  1,  1, 1, 1],\n    [  1,   2,   3,  4,  5, 6, 0],\n    [  2,   5,  10, 17, 26, 0, 0],\n    [  5,  15,  37, 77,  0, 0, 0],\n    [ 15,  52, 151,  0,  0, 0, 0],\n    [ 52, 203,   0,  0,  0, 0, 0],\n    [203,   0,   0,  0,  0, 0, 0]])\n    '
    d = zeros(m + 1)
    for i in range(m + 1):
        d[0, i] = 1
    for i in range(1, m + 1):
        for j in range(m):
            if j <= m - i:
                d[i, j] = j * d[i - 1, j] + d[i - 1, j + 1]
            else:
                d[i, j] = 0
    return d

def RGS_enum(m):
    if False:
        i = 10
        return i + 15
    '\n    RGS_enum computes the total number of restricted growth strings\n    possible for a superset of size m.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.partitions import RGS_enum\n    >>> from sympy.combinatorics import Partition\n    >>> RGS_enum(4)\n    15\n    >>> RGS_enum(5)\n    52\n    >>> RGS_enum(6)\n    203\n\n    We can check that the enumeration is correct by actually generating\n    the partitions. Here, the 15 partitions of 4 items are generated:\n\n    >>> a = Partition(list(range(4)))\n    >>> s = set()\n    >>> for i in range(20):\n    ...     s.add(a)\n    ...     a += 1\n    ...\n    >>> assert len(s) == 15\n\n    '
    if m < 1:
        return 0
    elif m == 1:
        return 1
    else:
        return bell(m)

def RGS_unrank(rank, m):
    if False:
        for i in range(10):
            print('nop')
    '\n    Gives the unranked restricted growth string for a given\n    superset size.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.partitions import RGS_unrank\n    >>> RGS_unrank(14, 4)\n    [0, 1, 2, 3]\n    >>> RGS_unrank(0, 4)\n    [0, 0, 0, 0]\n    '
    if m < 1:
        raise ValueError('The superset size must be >= 1')
    if rank < 0 or RGS_enum(m) <= rank:
        raise ValueError('Invalid arguments')
    L = [1] * (m + 1)
    j = 1
    D = RGS_generalized(m)
    for i in range(2, m + 1):
        v = D[m - i, j]
        cr = j * v
        if cr <= rank:
            L[i] = j + 1
            rank -= cr
            j += 1
        else:
            L[i] = int(rank / v + 1)
            rank %= v
    return [x - 1 for x in L[1:]]

def RGS_rank(rgs):
    if False:
        return 10
    '\n    Computes the rank of a restricted growth string.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.partitions import RGS_rank, RGS_unrank\n    >>> RGS_rank([0, 1, 2, 1, 3])\n    42\n    >>> RGS_rank(RGS_unrank(4, 7))\n    4\n    '
    rgs_size = len(rgs)
    rank = 0
    D = RGS_generalized(rgs_size)
    for i in range(1, rgs_size):
        n = len(rgs[i + 1:])
        m = max(rgs[0:i])
        rank += D[n, m + 1] * rgs[i]
    return rank