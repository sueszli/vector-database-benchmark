from sympy.core import Basic, Integer
import random

class GrayCode(Basic):
    """
    A Gray code is essentially a Hamiltonian walk on
    a n-dimensional cube with edge length of one.
    The vertices of the cube are represented by vectors
    whose values are binary. The Hamilton walk visits
    each vertex exactly once. The Gray code for a 3d
    cube is ['000','100','110','010','011','111','101',
    '001'].

    A Gray code solves the problem of sequentially
    generating all possible subsets of n objects in such
    a way that each subset is obtained from the previous
    one by either deleting or adding a single object.
    In the above example, 1 indicates that the object is
    present, and 0 indicates that its absent.

    Gray codes have applications in statistics as well when
    we want to compute various statistics related to subsets
    in an efficient manner.

    Examples
    ========

    >>> from sympy.combinatorics import GrayCode
    >>> a = GrayCode(3)
    >>> list(a.generate_gray())
    ['000', '001', '011', '010', '110', '111', '101', '100']
    >>> a = GrayCode(4)
    >>> list(a.generate_gray())
    ['0000', '0001', '0011', '0010', '0110', '0111', '0101', '0100',     '1100', '1101', '1111', '1110', '1010', '1011', '1001', '1000']

    References
    ==========

    .. [1] Nijenhuis,A. and Wilf,H.S.(1978).
           Combinatorial Algorithms. Academic Press.
    .. [2] Knuth, D. (2011). The Art of Computer Programming, Vol 4
           Addison Wesley


    """
    _skip = False
    _current = 0
    _rank = None

    def __new__(cls, n, *args, **kw_args):
        if False:
            while True:
                i = 10
        "\n        Default constructor.\n\n        It takes a single argument ``n`` which gives the dimension of the Gray\n        code. The starting Gray code string (``start``) or the starting ``rank``\n        may also be given; the default is to start at rank = 0 ('0...0').\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import GrayCode\n        >>> a = GrayCode(3)\n        >>> a\n        GrayCode(3)\n        >>> a.n\n        3\n\n        >>> a = GrayCode(3, start='100')\n        >>> a.current\n        '100'\n\n        >>> a = GrayCode(4, rank=4)\n        >>> a.current\n        '0110'\n        >>> a.rank\n        4\n\n        "
        if n < 1 or int(n) != n:
            raise ValueError('Gray code dimension must be a positive integer, not %i' % n)
        n = Integer(n)
        args = (n,) + args
        obj = Basic.__new__(cls, *args)
        if 'start' in kw_args:
            obj._current = kw_args['start']
            if len(obj._current) > n:
                raise ValueError('Gray code start has length %i but should not be greater than %i' % (len(obj._current), n))
        elif 'rank' in kw_args:
            if int(kw_args['rank']) != kw_args['rank']:
                raise ValueError('Gray code rank must be a positive integer, not %i' % kw_args['rank'])
            obj._rank = int(kw_args['rank']) % obj.selections
            obj._current = obj.unrank(n, obj._rank)
        return obj

    def next(self, delta=1):
        if False:
            i = 10
            return i + 15
        "\n        Returns the Gray code a distance ``delta`` (default = 1) from the\n        current value in canonical order.\n\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import GrayCode\n        >>> a = GrayCode(3, start='110')\n        >>> a.next().current\n        '111'\n        >>> a.next(-1).current\n        '010'\n        "
        return GrayCode(self.n, rank=(self.rank + delta) % self.selections)

    @property
    def selections(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the number of bit vectors in the Gray code.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import GrayCode\n        >>> a = GrayCode(3)\n        >>> a.selections\n        8\n        '
        return 2 ** self.n

    @property
    def n(self):
        if False:
            print('Hello World!')
        '\n        Returns the dimension of the Gray code.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import GrayCode\n        >>> a = GrayCode(5)\n        >>> a.n\n        5\n        '
        return self.args[0]

    def generate_gray(self, **hints):
        if False:
            return 10
        "\n        Generates the sequence of bit vectors of a Gray Code.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import GrayCode\n        >>> a = GrayCode(3)\n        >>> list(a.generate_gray())\n        ['000', '001', '011', '010', '110', '111', '101', '100']\n        >>> list(a.generate_gray(start='011'))\n        ['011', '010', '110', '111', '101', '100']\n        >>> list(a.generate_gray(rank=4))\n        ['110', '111', '101', '100']\n\n        See Also\n        ========\n\n        skip\n\n        References\n        ==========\n\n        .. [1] Knuth, D. (2011). The Art of Computer Programming,\n               Vol 4, Addison Wesley\n\n        "
        bits = self.n
        start = None
        if 'start' in hints:
            start = hints['start']
        elif 'rank' in hints:
            start = GrayCode.unrank(self.n, hints['rank'])
        if start is not None:
            self._current = start
        current = self.current
        graycode_bin = gray_to_bin(current)
        if len(graycode_bin) > self.n:
            raise ValueError('Gray code start has length %i but should not be greater than %i' % (len(graycode_bin), bits))
        self._current = int(current, 2)
        graycode_int = int(''.join(graycode_bin), 2)
        for i in range(graycode_int, 1 << bits):
            if self._skip:
                self._skip = False
            else:
                yield self.current
            bbtc = i ^ i + 1
            gbtc = bbtc ^ bbtc >> 1
            self._current = self._current ^ gbtc
        self._current = 0

    def skip(self):
        if False:
            while True:
                i = 10
        "\n        Skips the bit generation.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import GrayCode\n        >>> a = GrayCode(3)\n        >>> for i in a.generate_gray():\n        ...     if i == '010':\n        ...         a.skip()\n        ...     print(i)\n        ...\n        000\n        001\n        011\n        010\n        111\n        101\n        100\n\n        See Also\n        ========\n\n        generate_gray\n        "
        self._skip = True

    @property
    def rank(self):
        if False:
            while True:
                i = 10
        "\n        Ranks the Gray code.\n\n        A ranking algorithm determines the position (or rank)\n        of a combinatorial object among all the objects w.r.t.\n        a given order. For example, the 4 bit binary reflected\n        Gray code (BRGC) '0101' has a rank of 6 as it appears in\n        the 6th position in the canonical ordering of the family\n        of 4 bit Gray codes.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import GrayCode\n        >>> a = GrayCode(3)\n        >>> list(a.generate_gray())\n        ['000', '001', '011', '010', '110', '111', '101', '100']\n        >>> GrayCode(3, start='100').rank\n        7\n        >>> GrayCode(3, rank=7).current\n        '100'\n\n        See Also\n        ========\n\n        unrank\n\n        References\n        ==========\n\n        .. [1] https://web.archive.org/web/20200224064753/http://statweb.stanford.edu/~susan/courses/s208/node12.html\n\n        "
        if self._rank is None:
            self._rank = int(gray_to_bin(self.current), 2)
        return self._rank

    @property
    def current(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns the currently referenced Gray code as a bit string.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import GrayCode\n        >>> GrayCode(3, start='100').current\n        '100'\n        "
        rv = self._current or '0'
        if not isinstance(rv, str):
            rv = bin(rv)[2:]
        return rv.rjust(self.n, '0')

    @classmethod
    def unrank(self, n, rank):
        if False:
            while True:
                i = 10
        "\n        Unranks an n-bit sized Gray code of rank k. This method exists\n        so that a derivative GrayCode class can define its own code of\n        a given rank.\n\n        The string here is generated in reverse order to allow for tail-call\n        optimization.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import GrayCode\n        >>> GrayCode(5, rank=3).current\n        '00010'\n        >>> GrayCode.unrank(5, 3)\n        '00010'\n\n        See Also\n        ========\n\n        rank\n        "

        def _unrank(k, n):
            if False:
                for i in range(10):
                    print('nop')
            if n == 1:
                return str(k % 2)
            m = 2 ** (n - 1)
            if k < m:
                return '0' + _unrank(k, n - 1)
            return '1' + _unrank(m - k % m - 1, n - 1)
        return _unrank(rank, n)

def random_bitstring(n):
    if False:
        while True:
            i = 10
    '\n    Generates a random bitlist of length n.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.graycode import random_bitstring\n    >>> random_bitstring(3) # doctest: +SKIP\n    100\n    '
    return ''.join([random.choice('01') for i in range(n)])

def gray_to_bin(bin_list):
    if False:
        i = 10
        return i + 15
    "\n    Convert from Gray coding to binary coding.\n\n    We assume big endian encoding.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.graycode import gray_to_bin\n    >>> gray_to_bin('100')\n    '111'\n\n    See Also\n    ========\n\n    bin_to_gray\n    "
    b = [bin_list[0]]
    for i in range(1, len(bin_list)):
        b += str(int(b[i - 1] != bin_list[i]))
    return ''.join(b)

def bin_to_gray(bin_list):
    if False:
        print('Hello World!')
    "\n    Convert from binary coding to gray coding.\n\n    We assume big endian encoding.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.graycode import bin_to_gray\n    >>> bin_to_gray('111')\n    '100'\n\n    See Also\n    ========\n\n    gray_to_bin\n    "
    b = [bin_list[0]]
    for i in range(1, len(bin_list)):
        b += str(int(bin_list[i]) ^ int(bin_list[i - 1]))
    return ''.join(b)

def get_subset_from_bitstring(super_set, bitstring):
    if False:
        return 10
    "\n    Gets the subset defined by the bitstring.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.graycode import get_subset_from_bitstring\n    >>> get_subset_from_bitstring(['a', 'b', 'c', 'd'], '0011')\n    ['c', 'd']\n    >>> get_subset_from_bitstring(['c', 'a', 'c', 'c'], '1100')\n    ['c', 'a']\n\n    See Also\n    ========\n\n    graycode_subsets\n    "
    if len(super_set) != len(bitstring):
        raise ValueError('The sizes of the lists are not equal')
    return [super_set[i] for (i, j) in enumerate(bitstring) if bitstring[i] == '1']

def graycode_subsets(gray_code_set):
    if False:
        i = 10
        return i + 15
    "\n    Generates the subsets as enumerated by a Gray code.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.graycode import graycode_subsets\n    >>> list(graycode_subsets(['a', 'b', 'c']))\n    [[], ['c'], ['b', 'c'], ['b'], ['a', 'b'], ['a', 'b', 'c'],     ['a', 'c'], ['a']]\n    >>> list(graycode_subsets(['a', 'b', 'c', 'c']))\n    [[], ['c'], ['c', 'c'], ['c'], ['b', 'c'], ['b', 'c', 'c'],     ['b', 'c'], ['b'], ['a', 'b'], ['a', 'b', 'c'], ['a', 'b', 'c', 'c'],     ['a', 'b', 'c'], ['a', 'c'], ['a', 'c', 'c'], ['a', 'c'], ['a']]\n\n    See Also\n    ========\n\n    get_subset_from_bitstring\n    "
    for bitstring in list(GrayCode(len(gray_code_set)).generate_gray()):
        yield get_subset_from_bitstring(gray_code_set, bitstring)