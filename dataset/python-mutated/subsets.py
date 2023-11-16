from itertools import combinations
from sympy.combinatorics.graycode import GrayCode

class Subset:
    """
    Represents a basic subset object.

    Explanation
    ===========

    We generate subsets using essentially two techniques,
    binary enumeration and lexicographic enumeration.
    The Subset class takes two arguments, the first one
    describes the initial subset to consider and the second
    describes the superset.

    Examples
    ========

    >>> from sympy.combinatorics import Subset
    >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
    >>> a.next_binary().subset
    ['b']
    >>> a.prev_binary().subset
    ['c']
    """
    _rank_binary = None
    _rank_lex = None
    _rank_graycode = None
    _subset = None
    _superset = None

    def __new__(cls, subset, superset):
        if False:
            while True:
                i = 10
        "\n        Default constructor.\n\n        It takes the ``subset`` and its ``superset`` as its parameters.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])\n        >>> a.subset\n        ['c', 'd']\n        >>> a.superset\n        ['a', 'b', 'c', 'd']\n        >>> a.size\n        2\n        "
        if len(subset) > len(superset):
            raise ValueError('Invalid arguments have been provided. The superset must be larger than the subset.')
        for elem in subset:
            if elem not in superset:
                raise ValueError('The superset provided is invalid as it does not contain the element {}'.format(elem))
        obj = object.__new__(cls)
        obj._subset = subset
        obj._superset = superset
        return obj

    def __eq__(self, other):
        if False:
            print('Hello World!')
        'Return a boolean indicating whether a == b on the basis of\n        whether both objects are of the class Subset and if the values\n        of the subset and superset attributes are the same.\n        '
        if not isinstance(other, Subset):
            return NotImplemented
        return self.subset == other.subset and self.superset == other.superset

    def iterate_binary(self, k):
        if False:
            while True:
                i = 10
        "\n        This is a helper function. It iterates over the\n        binary subsets by ``k`` steps. This variable can be\n        both positive or negative.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])\n        >>> a.iterate_binary(-2).subset\n        ['d']\n        >>> a = Subset(['a', 'b', 'c'], ['a', 'b', 'c', 'd'])\n        >>> a.iterate_binary(2).subset\n        []\n\n        See Also\n        ========\n\n        next_binary, prev_binary\n        "
        bin_list = Subset.bitlist_from_subset(self.subset, self.superset)
        n = (int(''.join(bin_list), 2) + k) % 2 ** self.superset_size
        bits = bin(n)[2:].rjust(self.superset_size, '0')
        return Subset.subset_from_bitlist(self.superset, bits)

    def next_binary(self):
        if False:
            print('Hello World!')
        "\n        Generates the next binary ordered subset.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])\n        >>> a.next_binary().subset\n        ['b']\n        >>> a = Subset(['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd'])\n        >>> a.next_binary().subset\n        []\n\n        See Also\n        ========\n\n        prev_binary, iterate_binary\n        "
        return self.iterate_binary(1)

    def prev_binary(self):
        if False:
            return 10
        "\n        Generates the previous binary ordered subset.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> a = Subset([], ['a', 'b', 'c', 'd'])\n        >>> a.prev_binary().subset\n        ['a', 'b', 'c', 'd']\n        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])\n        >>> a.prev_binary().subset\n        ['c']\n\n        See Also\n        ========\n\n        next_binary, iterate_binary\n        "
        return self.iterate_binary(-1)

    def next_lexicographic(self):
        if False:
            return 10
        "\n        Generates the next lexicographically ordered subset.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])\n        >>> a.next_lexicographic().subset\n        ['d']\n        >>> a = Subset(['d'], ['a', 'b', 'c', 'd'])\n        >>> a.next_lexicographic().subset\n        []\n\n        See Also\n        ========\n\n        prev_lexicographic\n        "
        i = self.superset_size - 1
        indices = Subset.subset_indices(self.subset, self.superset)
        if i in indices:
            if i - 1 in indices:
                indices.remove(i - 1)
            else:
                indices.remove(i)
                i = i - 1
                while i >= 0 and i not in indices:
                    i = i - 1
                if i >= 0:
                    indices.remove(i)
                    indices.append(i + 1)
        else:
            while i not in indices and i >= 0:
                i = i - 1
            indices.append(i + 1)
        ret_set = []
        super_set = self.superset
        for i in indices:
            ret_set.append(super_set[i])
        return Subset(ret_set, super_set)

    def prev_lexicographic(self):
        if False:
            print('Hello World!')
        "\n        Generates the previous lexicographically ordered subset.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> a = Subset([], ['a', 'b', 'c', 'd'])\n        >>> a.prev_lexicographic().subset\n        ['d']\n        >>> a = Subset(['c','d'], ['a', 'b', 'c', 'd'])\n        >>> a.prev_lexicographic().subset\n        ['c']\n\n        See Also\n        ========\n\n        next_lexicographic\n        "
        i = self.superset_size - 1
        indices = Subset.subset_indices(self.subset, self.superset)
        while i >= 0 and i not in indices:
            i = i - 1
        if i == 0 or i - 1 in indices:
            indices.remove(i)
        else:
            if i >= 0:
                indices.remove(i)
                indices.append(i - 1)
            indices.append(self.superset_size - 1)
        ret_set = []
        super_set = self.superset
        for i in indices:
            ret_set.append(super_set[i])
        return Subset(ret_set, super_set)

    def iterate_graycode(self, k):
        if False:
            while True:
                i = 10
        '\n        Helper function used for prev_gray and next_gray.\n        It performs ``k`` step overs to get the respective Gray codes.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> a = Subset([1, 2, 3], [1, 2, 3, 4])\n        >>> a.iterate_graycode(3).subset\n        [1, 4]\n        >>> a.iterate_graycode(-2).subset\n        [1, 2, 4]\n\n        See Also\n        ========\n\n        next_gray, prev_gray\n        '
        unranked_code = GrayCode.unrank(self.superset_size, (self.rank_gray + k) % self.cardinality)
        return Subset.subset_from_bitlist(self.superset, unranked_code)

    def next_gray(self):
        if False:
            while True:
                i = 10
        '\n        Generates the next Gray code ordered subset.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> a = Subset([1, 2, 3], [1, 2, 3, 4])\n        >>> a.next_gray().subset\n        [1, 3]\n\n        See Also\n        ========\n\n        iterate_graycode, prev_gray\n        '
        return self.iterate_graycode(1)

    def prev_gray(self):
        if False:
            while True:
                i = 10
        '\n        Generates the previous Gray code ordered subset.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> a = Subset([2, 3, 4], [1, 2, 3, 4, 5])\n        >>> a.prev_gray().subset\n        [2, 3, 4, 5]\n\n        See Also\n        ========\n\n        iterate_graycode, next_gray\n        '
        return self.iterate_graycode(-1)

    @property
    def rank_binary(self):
        if False:
            i = 10
            return i + 15
        "\n        Computes the binary ordered rank.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> a = Subset([], ['a','b','c','d'])\n        >>> a.rank_binary\n        0\n        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])\n        >>> a.rank_binary\n        3\n\n        See Also\n        ========\n\n        iterate_binary, unrank_binary\n        "
        if self._rank_binary is None:
            self._rank_binary = int(''.join(Subset.bitlist_from_subset(self.subset, self.superset)), 2)
        return self._rank_binary

    @property
    def rank_lexicographic(self):
        if False:
            return 10
        "\n        Computes the lexicographic ranking of the subset.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])\n        >>> a.rank_lexicographic\n        14\n        >>> a = Subset([2, 4, 5], [1, 2, 3, 4, 5, 6])\n        >>> a.rank_lexicographic\n        43\n        "
        if self._rank_lex is None:

            def _ranklex(self, subset_index, i, n):
                if False:
                    return 10
                if subset_index == [] or i > n:
                    return 0
                if i in subset_index:
                    subset_index.remove(i)
                    return 1 + _ranklex(self, subset_index, i + 1, n)
                return 2 ** (n - i - 1) + _ranklex(self, subset_index, i + 1, n)
            indices = Subset.subset_indices(self.subset, self.superset)
            self._rank_lex = _ranklex(self, indices, 0, self.superset_size)
        return self._rank_lex

    @property
    def rank_gray(self):
        if False:
            return 10
        "\n        Computes the Gray code ranking of the subset.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> a = Subset(['c','d'], ['a','b','c','d'])\n        >>> a.rank_gray\n        2\n        >>> a = Subset([2, 4, 5], [1, 2, 3, 4, 5, 6])\n        >>> a.rank_gray\n        27\n\n        See Also\n        ========\n\n        iterate_graycode, unrank_gray\n        "
        if self._rank_graycode is None:
            bits = Subset.bitlist_from_subset(self.subset, self.superset)
            self._rank_graycode = GrayCode(len(bits), start=bits).rank
        return self._rank_graycode

    @property
    def subset(self):
        if False:
            while True:
                i = 10
        "\n        Gets the subset represented by the current instance.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])\n        >>> a.subset\n        ['c', 'd']\n\n        See Also\n        ========\n\n        superset, size, superset_size, cardinality\n        "
        return self._subset

    @property
    def size(self):
        if False:
            while True:
                i = 10
        "\n        Gets the size of the subset.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])\n        >>> a.size\n        2\n\n        See Also\n        ========\n\n        subset, superset, superset_size, cardinality\n        "
        return len(self.subset)

    @property
    def superset(self):
        if False:
            return 10
        "\n        Gets the superset of the subset.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])\n        >>> a.superset\n        ['a', 'b', 'c', 'd']\n\n        See Also\n        ========\n\n        subset, size, superset_size, cardinality\n        "
        return self._superset

    @property
    def superset_size(self):
        if False:
            while True:
                i = 10
        "\n        Returns the size of the superset.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])\n        >>> a.superset_size\n        4\n\n        See Also\n        ========\n\n        subset, superset, size, cardinality\n        "
        return len(self.superset)

    @property
    def cardinality(self):
        if False:
            print('Hello World!')
        "\n        Returns the number of all possible subsets.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])\n        >>> a.cardinality\n        16\n\n        See Also\n        ========\n\n        subset, superset, size, superset_size\n        "
        return 2 ** self.superset_size

    @classmethod
    def subset_from_bitlist(self, super_set, bitlist):
        if False:
            for i in range(10):
                print('nop')
        "\n        Gets the subset defined by the bitlist.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> Subset.subset_from_bitlist(['a', 'b', 'c', 'd'], '0011').subset\n        ['c', 'd']\n\n        See Also\n        ========\n\n        bitlist_from_subset\n        "
        if len(super_set) != len(bitlist):
            raise ValueError('The sizes of the lists are not equal')
        ret_set = []
        for i in range(len(bitlist)):
            if bitlist[i] == '1':
                ret_set.append(super_set[i])
        return Subset(ret_set, super_set)

    @classmethod
    def bitlist_from_subset(self, subset, superset):
        if False:
            while True:
                i = 10
        "\n        Gets the bitlist corresponding to a subset.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> Subset.bitlist_from_subset(['c', 'd'], ['a', 'b', 'c', 'd'])\n        '0011'\n\n        See Also\n        ========\n\n        subset_from_bitlist\n        "
        bitlist = ['0'] * len(superset)
        if isinstance(subset, Subset):
            subset = subset.subset
        for i in Subset.subset_indices(subset, superset):
            bitlist[i] = '1'
        return ''.join(bitlist)

    @classmethod
    def unrank_binary(self, rank, superset):
        if False:
            i = 10
            return i + 15
        "\n        Gets the binary ordered subset of the specified rank.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> Subset.unrank_binary(4, ['a', 'b', 'c', 'd']).subset\n        ['b']\n\n        See Also\n        ========\n\n        iterate_binary, rank_binary\n        "
        bits = bin(rank)[2:].rjust(len(superset), '0')
        return Subset.subset_from_bitlist(superset, bits)

    @classmethod
    def unrank_gray(self, rank, superset):
        if False:
            while True:
                i = 10
        "\n        Gets the Gray code ordered subset of the specified rank.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Subset\n        >>> Subset.unrank_gray(4, ['a', 'b', 'c']).subset\n        ['a', 'b']\n        >>> Subset.unrank_gray(0, ['a', 'b', 'c']).subset\n        []\n\n        See Also\n        ========\n\n        iterate_graycode, rank_gray\n        "
        graycode_bitlist = GrayCode.unrank(len(superset), rank)
        return Subset.subset_from_bitlist(superset, graycode_bitlist)

    @classmethod
    def subset_indices(self, subset, superset):
        if False:
            i = 10
            return i + 15
        'Return indices of subset in superset in a list; the list is empty\n        if all elements of ``subset`` are not in ``superset``.\n\n        Examples\n        ========\n\n            >>> from sympy.combinatorics import Subset\n            >>> superset = [1, 3, 2, 5, 4]\n            >>> Subset.subset_indices([3, 2, 1], superset)\n            [1, 2, 0]\n            >>> Subset.subset_indices([1, 6], superset)\n            []\n            >>> Subset.subset_indices([], superset)\n            []\n\n        '
        (a, b) = (superset, subset)
        sb = set(b)
        d = {}
        for (i, ai) in enumerate(a):
            if ai in sb:
                d[ai] = i
                sb.remove(ai)
                if not sb:
                    break
        else:
            return []
        return [d[bi] for bi in b]

def ksubsets(superset, k):
    if False:
        for i in range(10):
            print('nop')
    '\n    Finds the subsets of size ``k`` in lexicographic order.\n\n    This uses the itertools generator.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.subsets import ksubsets\n    >>> list(ksubsets([1, 2, 3], 2))\n    [(1, 2), (1, 3), (2, 3)]\n    >>> list(ksubsets([1, 2, 3, 4, 5], 2))\n    [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4),     (2, 5), (3, 4), (3, 5), (4, 5)]\n\n    See Also\n    ========\n\n    Subset\n    '
    return combinations(superset, k)