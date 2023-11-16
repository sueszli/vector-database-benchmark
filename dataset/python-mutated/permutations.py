import random
from collections import defaultdict
from collections.abc import Iterable
from functools import reduce
from sympy.core.parameters import global_parameters
from sympy.core.basic import Atom
from sympy.core.expr import Expr
from sympy.core.numbers import int_valued
from sympy.core.numbers import Integer
from sympy.core.sympify import _sympify
from sympy.matrices import zeros
from sympy.polys.polytools import lcm
from sympy.printing.repr import srepr
from sympy.utilities.iterables import flatten, has_variety, minlex, has_dups, runs, is_sequence
from sympy.utilities.misc import as_int
from mpmath.libmp.libintmath import ifac
from sympy.multipledispatch import dispatch

def _af_rmul(a, b):
    if False:
        return 10
    '\n    Return the product b*a; input and output are array forms. The ith value\n    is a[b[i]].\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.permutations import _af_rmul, Permutation\n\n    >>> a, b = [1, 0, 2], [0, 2, 1]\n    >>> _af_rmul(a, b)\n    [1, 2, 0]\n    >>> [a[b[i]] for i in range(3)]\n    [1, 2, 0]\n\n    This handles the operands in reverse order compared to the ``*`` operator:\n\n    >>> a = Permutation(a)\n    >>> b = Permutation(b)\n    >>> list(a*b)\n    [2, 0, 1]\n    >>> [b(a(i)) for i in range(3)]\n    [2, 0, 1]\n\n    See Also\n    ========\n\n    rmul, _af_rmuln\n    '
    return [a[i] for i in b]

def _af_rmuln(*abc):
    if False:
        i = 10
        return i + 15
    '\n    Given [a, b, c, ...] return the product of ...*c*b*a using array forms.\n    The ith value is a[b[c[i]]].\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.permutations import _af_rmul, Permutation\n\n    >>> a, b = [1, 0, 2], [0, 2, 1]\n    >>> _af_rmul(a, b)\n    [1, 2, 0]\n    >>> [a[b[i]] for i in range(3)]\n    [1, 2, 0]\n\n    This handles the operands in reverse order compared to the ``*`` operator:\n\n    >>> a = Permutation(a); b = Permutation(b)\n    >>> list(a*b)\n    [2, 0, 1]\n    >>> [b(a(i)) for i in range(3)]\n    [2, 0, 1]\n\n    See Also\n    ========\n\n    rmul, _af_rmul\n    '
    a = abc
    m = len(a)
    if m == 3:
        (p0, p1, p2) = a
        return [p0[p1[i]] for i in p2]
    if m == 4:
        (p0, p1, p2, p3) = a
        return [p0[p1[p2[i]]] for i in p3]
    if m == 5:
        (p0, p1, p2, p3, p4) = a
        return [p0[p1[p2[p3[i]]]] for i in p4]
    if m == 6:
        (p0, p1, p2, p3, p4, p5) = a
        return [p0[p1[p2[p3[p4[i]]]]] for i in p5]
    if m == 7:
        (p0, p1, p2, p3, p4, p5, p6) = a
        return [p0[p1[p2[p3[p4[p5[i]]]]]] for i in p6]
    if m == 8:
        (p0, p1, p2, p3, p4, p5, p6, p7) = a
        return [p0[p1[p2[p3[p4[p5[p6[i]]]]]]] for i in p7]
    if m == 1:
        return a[0][:]
    if m == 2:
        (a, b) = a
        return [a[i] for i in b]
    if m == 0:
        raise ValueError('String must not be empty')
    p0 = _af_rmuln(*a[:m // 2])
    p1 = _af_rmuln(*a[m // 2:])
    return [p0[i] for i in p1]

def _af_parity(pi):
    if False:
        i = 10
        return i + 15
    '\n    Computes the parity of a permutation in array form.\n\n    Explanation\n    ===========\n\n    The parity of a permutation reflects the parity of the\n    number of inversions in the permutation, i.e., the\n    number of pairs of x and y such that x > y but p[x] < p[y].\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.permutations import _af_parity\n    >>> _af_parity([0, 1, 2, 3])\n    0\n    >>> _af_parity([3, 2, 0, 1])\n    1\n\n    See Also\n    ========\n\n    Permutation\n    '
    n = len(pi)
    a = [0] * n
    c = 0
    for j in range(n):
        if a[j] == 0:
            c += 1
            a[j] = 1
            i = j
            while pi[i] != j:
                i = pi[i]
                a[i] = 1
    return (n - c) % 2

def _af_invert(a):
    if False:
        return 10
    '\n    Finds the inverse, ~A, of a permutation, A, given in array form.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.permutations import _af_invert, _af_rmul\n    >>> A = [1, 2, 0, 3]\n    >>> _af_invert(A)\n    [2, 0, 1, 3]\n    >>> _af_rmul(_, A)\n    [0, 1, 2, 3]\n\n    See Also\n    ========\n\n    Permutation, __invert__\n    '
    inv_form = [0] * len(a)
    for (i, ai) in enumerate(a):
        inv_form[ai] = i
    return inv_form

def _af_pow(a, n):
    if False:
        while True:
            i = 10
    '\n    Routine for finding powers of a permutation.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics import Permutation\n    >>> from sympy.combinatorics.permutations import _af_pow\n    >>> p = Permutation([2, 0, 3, 1])\n    >>> p.order()\n    4\n    >>> _af_pow(p._array_form, 4)\n    [0, 1, 2, 3]\n    '
    if n == 0:
        return list(range(len(a)))
    if n < 0:
        return _af_pow(_af_invert(a), -n)
    if n == 1:
        return a[:]
    elif n == 2:
        b = [a[i] for i in a]
    elif n == 3:
        b = [a[a[i]] for i in a]
    elif n == 4:
        b = [a[a[a[i]]] for i in a]
    else:
        b = list(range(len(a)))
        while 1:
            if n & 1:
                b = [b[i] for i in a]
                n -= 1
                if not n:
                    break
            if n % 4 == 0:
                a = [a[a[a[i]]] for i in a]
                n = n // 4
            elif n % 2 == 0:
                a = [a[i] for i in a]
                n = n // 2
    return b

def _af_commutes_with(a, b):
    if False:
        print('Hello World!')
    '\n    Checks if the two permutations with array forms\n    given by ``a`` and ``b`` commute.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.permutations import _af_commutes_with\n    >>> _af_commutes_with([1, 2, 0], [0, 2, 1])\n    False\n\n    See Also\n    ========\n\n    Permutation, commutes_with\n    '
    return not any((a[b[i]] != b[a[i]] for i in range(len(a) - 1)))

class Cycle(dict):
    """
    Wrapper around dict which provides the functionality of a disjoint cycle.

    Explanation
    ===========

    A cycle shows the rule to use to move subsets of elements to obtain
    a permutation. The Cycle class is more flexible than Permutation in
    that 1) all elements need not be present in order to investigate how
    multiple cycles act in sequence and 2) it can contain singletons:

    >>> from sympy.combinatorics.permutations import Perm, Cycle

    A Cycle will automatically parse a cycle given as a tuple on the rhs:

    >>> Cycle(1, 2)(2, 3)
    (1 3 2)

    The identity cycle, Cycle(), can be used to start a product:

    >>> Cycle()(1, 2)(2, 3)
    (1 3 2)

    The array form of a Cycle can be obtained by calling the list
    method (or passing it to the list function) and all elements from
    0 will be shown:

    >>> a = Cycle(1, 2)
    >>> a.list()
    [0, 2, 1]
    >>> list(a)
    [0, 2, 1]

    If a larger (or smaller) range is desired use the list method and
    provide the desired size -- but the Cycle cannot be truncated to
    a size smaller than the largest element that is out of place:

    >>> b = Cycle(2, 4)(1, 2)(3, 1, 4)(1, 3)
    >>> b.list()
    [0, 2, 1, 3, 4]
    >>> b.list(b.size + 1)
    [0, 2, 1, 3, 4, 5]
    >>> b.list(-1)
    [0, 2, 1]

    Singletons are not shown when printing with one exception: the largest
    element is always shown -- as a singleton if necessary:

    >>> Cycle(1, 4, 10)(4, 5)
    (1 5 4 10)
    >>> Cycle(1, 2)(4)(5)(10)
    (1 2)(10)

    The array form can be used to instantiate a Permutation so other
    properties of the permutation can be investigated:

    >>> Perm(Cycle(1, 2)(3, 4).list()).transpositions()
    [(1, 2), (3, 4)]

    Notes
    =====

    The underlying structure of the Cycle is a dictionary and although
    the __iter__ method has been redefined to give the array form of the
    cycle, the underlying dictionary items are still available with the
    such methods as items():

    >>> list(Cycle(1, 2).items())
    [(1, 2), (2, 1)]

    See Also
    ========

    Permutation
    """

    def __missing__(self, arg):
        if False:
            i = 10
            return i + 15
        'Enter arg into dictionary and return arg.'
        return as_int(arg)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        yield from self.list()

    def __call__(self, *other):
        if False:
            print('Hello World!')
        'Return product of cycles processed from R to L.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Cycle\n        >>> Cycle(1, 2)(2, 3)\n        (1 3 2)\n\n        An instance of a Cycle will automatically parse list-like\n        objects and Permutations that are on the right. It is more\n        flexible than the Permutation in that all elements need not\n        be present:\n\n        >>> a = Cycle(1, 2)\n        >>> a(2, 3)\n        (1 3 2)\n        >>> a(2, 3)(4, 5)\n        (1 3 2)(4 5)\n\n        '
        rv = Cycle(*other)
        for (k, v) in zip(list(self.keys()), [rv[self[k]] for k in self.keys()]):
            rv[k] = v
        return rv

    def list(self, size=None):
        if False:
            while True:
                i = 10
        'Return the cycles as an explicit list starting from 0 up\n        to the greater of the largest value in the cycles and size.\n\n        Truncation of trailing unmoved items will occur when size\n        is less than the maximum element in the cycle; if this is\n        desired, setting ``size=-1`` will guarantee such trimming.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Cycle\n        >>> p = Cycle(2, 3)(4, 5)\n        >>> p.list()\n        [0, 1, 3, 2, 5, 4]\n        >>> p.list(10)\n        [0, 1, 3, 2, 5, 4, 6, 7, 8, 9]\n\n        Passing a length too small will trim trailing, unchanged elements\n        in the permutation:\n\n        >>> Cycle(2, 4)(1, 2, 4).list(-1)\n        [0, 2, 1]\n        '
        if not self and size is None:
            raise ValueError('must give size for empty Cycle')
        if size is not None:
            big = max([i for i in self.keys() if self[i] != i] + [0])
            size = max(size, big + 1)
        else:
            size = self.size
        return [self[i] for i in range(size)]

    def __repr__(self):
        if False:
            while True:
                i = 10
        'We want it to print as a Cycle, not as a dict.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Cycle\n        >>> Cycle(1, 2)\n        (1 2)\n        >>> print(_)\n        (1 2)\n        >>> list(Cycle(1, 2).items())\n        [(1, 2), (2, 1)]\n        '
        if not self:
            return 'Cycle()'
        cycles = Permutation(self).cyclic_form
        s = ''.join((str(tuple(c)) for c in cycles))
        big = self.size - 1
        if not any((i == big for c in cycles for i in c)):
            s += '(%s)' % big
        return 'Cycle%s' % s

    def __str__(self):
        if False:
            while True:
                i = 10
        'We want it to be printed in a Cycle notation with no\n        comma in-between.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Cycle\n        >>> Cycle(1, 2)\n        (1 2)\n        >>> Cycle(1, 2, 4)(5, 6)\n        (1 2 4)(5 6)\n        '
        if not self:
            return '()'
        cycles = Permutation(self).cyclic_form
        s = ''.join((str(tuple(c)) for c in cycles))
        big = self.size - 1
        if not any((i == big for c in cycles for i in c)):
            s += '(%s)' % big
        s = s.replace(',', '')
        return s

    def __init__(self, *args):
        if False:
            while True:
                i = 10
        'Load up a Cycle instance with the values for the cycle.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Cycle\n        >>> Cycle(1, 2, 6)\n        (1 2 6)\n        '
        if not args:
            return
        if len(args) == 1:
            if isinstance(args[0], Permutation):
                for c in args[0].cyclic_form:
                    self.update(self(*c))
                return
            elif isinstance(args[0], Cycle):
                for (k, v) in args[0].items():
                    self[k] = v
                return
        args = [as_int(a) for a in args]
        if any((i < 0 for i in args)):
            raise ValueError('negative integers are not allowed in a cycle.')
        if has_dups(args):
            raise ValueError('All elements must be unique in a cycle.')
        for i in range(-len(args), 0):
            self[args[i]] = args[i + 1]

    @property
    def size(self):
        if False:
            for i in range(10):
                print('nop')
        if not self:
            return 0
        return max(self.keys()) + 1

    def copy(self):
        if False:
            return 10
        return Cycle(self)

class Permutation(Atom):
    """
    A permutation, alternatively known as an 'arrangement number' or 'ordering'
    is an arrangement of the elements of an ordered list into a one-to-one
    mapping with itself. The permutation of a given arrangement is given by
    indicating the positions of the elements after re-arrangement [2]_. For
    example, if one started with elements ``[x, y, a, b]`` (in that order) and
    they were reordered as ``[x, y, b, a]`` then the permutation would be
    ``[0, 1, 3, 2]``. Notice that (in SymPy) the first element is always referred
    to as 0 and the permutation uses the indices of the elements in the
    original ordering, not the elements ``(a, b, ...)`` themselves.

    >>> from sympy.combinatorics import Permutation
    >>> from sympy import init_printing
    >>> init_printing(perm_cyclic=False, pretty_print=False)

    Permutations Notation
    =====================

    Permutations are commonly represented in disjoint cycle or array forms.

    Array Notation and 2-line Form
    ------------------------------------

    In the 2-line form, the elements and their final positions are shown
    as a matrix with 2 rows:

    [0    1    2     ... n-1]
    [p(0) p(1) p(2)  ... p(n-1)]

    Since the first line is always ``range(n)``, where n is the size of p,
    it is sufficient to represent the permutation by the second line,
    referred to as the "array form" of the permutation. This is entered
    in brackets as the argument to the Permutation class:

    >>> p = Permutation([0, 2, 1]); p
    Permutation([0, 2, 1])

    Given i in range(p.size), the permutation maps i to i^p

    >>> [i^p for i in range(p.size)]
    [0, 2, 1]

    The composite of two permutations p*q means first apply p, then q, so
    i^(p*q) = (i^p)^q which is i^p^q according to Python precedence rules:

    >>> q = Permutation([2, 1, 0])
    >>> [i^p^q for i in range(3)]
    [2, 0, 1]
    >>> [i^(p*q) for i in range(3)]
    [2, 0, 1]

    One can use also the notation p(i) = i^p, but then the composition
    rule is (p*q)(i) = q(p(i)), not p(q(i)):

    >>> [(p*q)(i) for i in range(p.size)]
    [2, 0, 1]
    >>> [q(p(i)) for i in range(p.size)]
    [2, 0, 1]
    >>> [p(q(i)) for i in range(p.size)]
    [1, 2, 0]

    Disjoint Cycle Notation
    -----------------------

    In disjoint cycle notation, only the elements that have shifted are
    indicated.

    For example, [1, 3, 2, 0] can be represented as (0, 1, 3)(2).
    This can be understood from the 2 line format of the given permutation.
    In the 2-line form,
    [0    1    2   3]
    [1    3    2   0]

    The element in the 0th position is 1, so 0 -> 1. The element in the 1st
    position is three, so 1 -> 3. And the element in the third position is again
    0, so 3 -> 0. Thus, 0 -> 1 -> 3 -> 0, and 2 -> 2. Thus, this can be represented
    as 2 cycles: (0, 1, 3)(2).
    In common notation, singular cycles are not explicitly written as they can be
    inferred implicitly.

    Only the relative ordering of elements in a cycle matter:

    >>> Permutation(1,2,3) == Permutation(2,3,1) == Permutation(3,1,2)
    True

    The disjoint cycle notation is convenient when representing
    permutations that have several cycles in them:

    >>> Permutation(1, 2)(3, 5) == Permutation([[1, 2], [3, 5]])
    True

    It also provides some economy in entry when computing products of
    permutations that are written in disjoint cycle notation:

    >>> Permutation(1, 2)(1, 3)(2, 3)
    Permutation([0, 3, 2, 1])
    >>> _ == Permutation([[1, 2]])*Permutation([[1, 3]])*Permutation([[2, 3]])
    True

        Caution: when the cycles have common elements between them then the order
        in which the permutations are applied matters. This module applies
        the permutations from *left to right*.

        >>> Permutation(1, 2)(2, 3) == Permutation([(1, 2), (2, 3)])
        True
        >>> Permutation(1, 2)(2, 3).list()
        [0, 3, 1, 2]

        In the above case, (1,2) is computed before (2,3).
        As 0 -> 0, 0 -> 0, element in position 0 is 0.
        As 1 -> 2, 2 -> 3, element in position 1 is 3.
        As 2 -> 1, 1 -> 1, element in position 2 is 1.
        As 3 -> 3, 3 -> 2, element in position 3 is 2.

        If the first and second elements had been
        swapped first, followed by the swapping of the second
        and third, the result would have been [0, 2, 3, 1].
        If, you want to apply the cycles in the conventional
        right to left order, call the function with arguments in reverse order
        as demonstrated below:

        >>> Permutation([(1, 2), (2, 3)][::-1]).list()
        [0, 2, 3, 1]

    Entering a singleton in a permutation is a way to indicate the size of the
    permutation. The ``size`` keyword can also be used.

    Array-form entry:

    >>> Permutation([[1, 2], [9]])
    Permutation([0, 2, 1], size=10)
    >>> Permutation([[1, 2]], size=10)
    Permutation([0, 2, 1], size=10)

    Cyclic-form entry:

    >>> Permutation(1, 2, size=10)
    Permutation([0, 2, 1], size=10)
    >>> Permutation(9)(1, 2)
    Permutation([0, 2, 1], size=10)

    Caution: no singleton containing an element larger than the largest
    in any previous cycle can be entered. This is an important difference
    in how Permutation and Cycle handle the ``__call__`` syntax. A singleton
    argument at the start of a Permutation performs instantiation of the
    Permutation and is permitted:

    >>> Permutation(5)
    Permutation([], size=6)

    A singleton entered after instantiation is a call to the permutation
    -- a function call -- and if the argument is out of range it will
    trigger an error. For this reason, it is better to start the cycle
    with the singleton:

    The following fails because there is no element 3:

    >>> Permutation(1, 2)(3)
    Traceback (most recent call last):
    ...
    IndexError: list index out of range

    This is ok: only the call to an out of range singleton is prohibited;
    otherwise the permutation autosizes:

    >>> Permutation(3)(1, 2)
    Permutation([0, 2, 1, 3])
    >>> Permutation(1, 2)(3, 4) == Permutation(3, 4)(1, 2)
    True


    Equality testing
    ----------------

    The array forms must be the same in order for permutations to be equal:

    >>> Permutation([1, 0, 2, 3]) == Permutation([1, 0])
    False


    Identity Permutation
    --------------------

    The identity permutation is a permutation in which no element is out of
    place. It can be entered in a variety of ways. All the following create
    an identity permutation of size 4:

    >>> I = Permutation([0, 1, 2, 3])
    >>> all(p == I for p in [
    ... Permutation(3),
    ... Permutation(range(4)),
    ... Permutation([], size=4),
    ... Permutation(size=4)])
    True

    Watch out for entering the range *inside* a set of brackets (which is
    cycle notation):

    >>> I == Permutation([range(4)])
    False


    Permutation Printing
    ====================

    There are a few things to note about how Permutations are printed.

    .. deprecated:: 1.6

       Configuring Permutation printing by setting
       ``Permutation.print_cyclic`` is deprecated. Users should use the
       ``perm_cyclic`` flag to the printers, as described below.

    1) If you prefer one form (array or cycle) over another, you can set
    ``init_printing`` with the ``perm_cyclic`` flag.

    >>> from sympy import init_printing
    >>> p = Permutation(1, 2)(4, 5)(3, 4)
    >>> p
    Permutation([0, 2, 1, 4, 5, 3])

    >>> init_printing(perm_cyclic=True, pretty_print=False)
    >>> p
    (1 2)(3 4 5)

    2) Regardless of the setting, a list of elements in the array for cyclic
    form can be obtained and either of those can be copied and supplied as
    the argument to Permutation:

    >>> p.array_form
    [0, 2, 1, 4, 5, 3]
    >>> p.cyclic_form
    [[1, 2], [3, 4, 5]]
    >>> Permutation(_) == p
    True

    3) Printing is economical in that as little as possible is printed while
    retaining all information about the size of the permutation:

    >>> init_printing(perm_cyclic=False, pretty_print=False)
    >>> Permutation([1, 0, 2, 3])
    Permutation([1, 0, 2, 3])
    >>> Permutation([1, 0, 2, 3], size=20)
    Permutation([1, 0], size=20)
    >>> Permutation([1, 0, 2, 4, 3, 5, 6], size=20)
    Permutation([1, 0, 2, 4, 3], size=20)

    >>> p = Permutation([1, 0, 2, 3])
    >>> init_printing(perm_cyclic=True, pretty_print=False)
    >>> p
    (3)(0 1)
    >>> init_printing(perm_cyclic=False, pretty_print=False)

    The 2 was not printed but it is still there as can be seen with the
    array_form and size methods:

    >>> p.array_form
    [1, 0, 2, 3]
    >>> p.size
    4

    Short introduction to other methods
    ===================================

    The permutation can act as a bijective function, telling what element is
    located at a given position

    >>> q = Permutation([5, 2, 3, 4, 1, 0])
    >>> q.array_form[1] # the hard way
    2
    >>> q(1) # the easy way
    2
    >>> {i: q(i) for i in range(q.size)} # showing the bijection
    {0: 5, 1: 2, 2: 3, 3: 4, 4: 1, 5: 0}

    The full cyclic form (including singletons) can be obtained:

    >>> p.full_cyclic_form
    [[0, 1], [2], [3]]

    Any permutation can be factored into transpositions of pairs of elements:

    >>> Permutation([[1, 2], [3, 4, 5]]).transpositions()
    [(1, 2), (3, 5), (3, 4)]
    >>> Permutation.rmul(*[Permutation([ti], size=6) for ti in _]).cyclic_form
    [[1, 2], [3, 4, 5]]

    The number of permutations on a set of n elements is given by n! and is
    called the cardinality.

    >>> p.size
    4
    >>> p.cardinality
    24

    A given permutation has a rank among all the possible permutations of the
    same elements, but what that rank is depends on how the permutations are
    enumerated. (There are a number of different methods of doing so.) The
    lexicographic rank is given by the rank method and this rank is used to
    increment a permutation with addition/subtraction:

    >>> p.rank()
    6
    >>> p + 1
    Permutation([1, 0, 3, 2])
    >>> p.next_lex()
    Permutation([1, 0, 3, 2])
    >>> _.rank()
    7
    >>> p.unrank_lex(p.size, rank=7)
    Permutation([1, 0, 3, 2])

    The product of two permutations p and q is defined as their composition as
    functions, (p*q)(i) = q(p(i)) [6]_.

    >>> p = Permutation([1, 0, 2, 3])
    >>> q = Permutation([2, 3, 1, 0])
    >>> list(q*p)
    [2, 3, 0, 1]
    >>> list(p*q)
    [3, 2, 1, 0]
    >>> [q(p(i)) for i in range(p.size)]
    [3, 2, 1, 0]

    The permutation can be 'applied' to any list-like object, not only
    Permutations:

    >>> p(['zero', 'one', 'four', 'two'])
    ['one', 'zero', 'four', 'two']
    >>> p('zo42')
    ['o', 'z', '4', '2']

    If you have a list of arbitrary elements, the corresponding permutation
    can be found with the from_sequence method:

    >>> Permutation.from_sequence('SymPy')
    Permutation([1, 3, 2, 0, 4])

    Checking if a Permutation is contained in a Group
    =================================================

    Generally if you have a group of permutations G on n symbols, and
    you're checking if a permutation on less than n symbols is part
    of that group, the check will fail.

    Here is an example for n=5 and we check if the cycle
    (1,2,3) is in G:

    >>> from sympy import init_printing
    >>> init_printing(perm_cyclic=True, pretty_print=False)
    >>> from sympy.combinatorics import Cycle, Permutation
    >>> from sympy.combinatorics.perm_groups import PermutationGroup
    >>> G = PermutationGroup(Cycle(2, 3)(4, 5), Cycle(1, 2, 3, 4, 5))
    >>> p1 = Permutation(Cycle(2, 5, 3))
    >>> p2 = Permutation(Cycle(1, 2, 3))
    >>> a1 = Permutation(Cycle(1, 2, 3).list(6))
    >>> a2 = Permutation(Cycle(1, 2, 3)(5))
    >>> a3 = Permutation(Cycle(1, 2, 3),size=6)
    >>> for p in [p1,p2,a1,a2,a3]: p, G.contains(p)
    ((2 5 3), True)
    ((1 2 3), False)
    ((5)(1 2 3), True)
    ((5)(1 2 3), True)
    ((5)(1 2 3), True)

    The check for p2 above will fail.

    Checking if p1 is in G works because SymPy knows
    G is a group on 5 symbols, and p1 is also on 5 symbols
    (its largest element is 5).

    For ``a1``, the ``.list(6)`` call will extend the permutation to 5
    symbols, so the test will work as well. In the case of ``a2`` the
    permutation is being extended to 5 symbols by using a singleton,
    and in the case of ``a3`` it's extended through the constructor
    argument ``size=6``.

    There is another way to do this, which is to tell the ``contains``
    method that the number of symbols the group is on does not need to
    match perfectly the number of symbols for the permutation:

    >>> G.contains(p2,strict=False)
    True

    This can be via the ``strict`` argument to the ``contains`` method,
    and SymPy will try to extend the permutation on its own and then
    perform the containment check.

    See Also
    ========

    Cycle

    References
    ==========

    .. [1] Skiena, S. 'Permutations.' 1.1 in Implementing Discrete Mathematics
           Combinatorics and Graph Theory with Mathematica.  Reading, MA:
           Addison-Wesley, pp. 3-16, 1990.

    .. [2] Knuth, D. E. The Art of Computer Programming, Vol. 4: Combinatorial
           Algorithms, 1st ed. Reading, MA: Addison-Wesley, 2011.

    .. [3] Wendy Myrvold and Frank Ruskey. 2001. Ranking and unranking
           permutations in linear time. Inf. Process. Lett. 79, 6 (September 2001),
           281-284. DOI=10.1016/S0020-0190(01)00141-7

    .. [4] D. L. Kreher, D. R. Stinson 'Combinatorial Algorithms'
           CRC Press, 1999

    .. [5] Graham, R. L.; Knuth, D. E.; and Patashnik, O.
           Concrete Mathematics: A Foundation for Computer Science, 2nd ed.
           Reading, MA: Addison-Wesley, 1994.

    .. [6] https://en.wikipedia.org/w/index.php?oldid=499948155#Product_and_inverse

    .. [7] https://en.wikipedia.org/wiki/Lehmer_code

    """
    is_Permutation = True
    _array_form = None
    _cyclic_form = None
    _cycle_structure = None
    _size = None
    _rank = None

    def __new__(cls, *args, size=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Constructor for the Permutation object from a list or a\n        list of lists in which all elements of the permutation may\n        appear only once.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> from sympy import init_printing\n        >>> init_printing(perm_cyclic=False, pretty_print=False)\n\n        Permutations entered in array-form are left unaltered:\n\n        >>> Permutation([0, 2, 1])\n        Permutation([0, 2, 1])\n\n        Permutations entered in cyclic form are converted to array form;\n        singletons need not be entered, but can be entered to indicate the\n        largest element:\n\n        >>> Permutation([[4, 5, 6], [0, 1]])\n        Permutation([1, 0, 2, 3, 5, 6, 4])\n        >>> Permutation([[4, 5, 6], [0, 1], [19]])\n        Permutation([1, 0, 2, 3, 5, 6, 4], size=20)\n\n        All manipulation of permutations assumes that the smallest element\n        is 0 (in keeping with 0-based indexing in Python) so if the 0 is\n        missing when entering a permutation in array form, an error will be\n        raised:\n\n        >>> Permutation([2, 1])\n        Traceback (most recent call last):\n        ...\n        ValueError: Integers 0 through 2 must be present.\n\n        If a permutation is entered in cyclic form, it can be entered without\n        singletons and the ``size`` specified so those values can be filled\n        in, otherwise the array form will only extend to the maximum value\n        in the cycles:\n\n        >>> Permutation([[1, 4], [3, 5, 2]], size=10)\n        Permutation([0, 4, 3, 5, 1, 2], size=10)\n        >>> _.array_form\n        [0, 4, 3, 5, 1, 2, 6, 7, 8, 9]\n        '
        if size is not None:
            size = int(size)
        ok = True
        if not args:
            return cls._af_new(list(range(size or 0)))
        elif len(args) > 1:
            return cls._af_new(Cycle(*args).list(size))
        if len(args) == 1:
            a = args[0]
            if isinstance(a, cls):
                if size is None or size == a.size:
                    return a
                return cls(a.array_form, size=size)
            if isinstance(a, Cycle):
                return cls._af_new(a.list(size))
            if not is_sequence(a):
                if size is not None and a + 1 > size:
                    raise ValueError('size is too small when max is %s' % a)
                return cls._af_new(list(range(a + 1)))
            if has_variety((is_sequence(ai) for ai in a)):
                ok = False
        else:
            ok = False
        if not ok:
            raise ValueError('Permutation argument must be a list of ints, a list of lists, Permutation or Cycle.')
        args = list(args[0])
        is_cycle = args and is_sequence(args[0])
        if is_cycle:
            args = [[int(i) for i in c] for c in args]
        else:
            args = [int(i) for i in args]
        temp = flatten(args)
        if has_dups(temp) and (not is_cycle):
            raise ValueError('there were repeated elements.')
        temp = set(temp)
        if not is_cycle:
            if temp != set(range(len(temp))):
                raise ValueError('Integers 0 through %s must be present.' % max(temp))
            if size is not None and temp and (max(temp) + 1 > size):
                raise ValueError('max element should not exceed %s' % (size - 1))
        if is_cycle:
            c = Cycle()
            for ci in args:
                c = c(*ci)
            aform = c.list()
        else:
            aform = list(args)
        if size and size > len(aform):
            aform.extend(list(range(len(aform), size)))
        return cls._af_new(aform)

    @classmethod
    def _af_new(cls, perm):
        if False:
            i = 10
            return i + 15
        'A method to produce a Permutation object from a list;\n        the list is bound to the _array_form attribute, so it must\n        not be modified; this method is meant for internal use only;\n        the list ``a`` is supposed to be generated as a temporary value\n        in a method, so p = Perm._af_new(a) is the only object\n        to hold a reference to ``a``::\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.permutations import Perm\n        >>> from sympy import init_printing\n        >>> init_printing(perm_cyclic=False, pretty_print=False)\n        >>> a = [2, 1, 3, 0]\n        >>> p = Perm._af_new(a)\n        >>> p\n        Permutation([2, 1, 3, 0])\n\n        '
        p = super().__new__(cls)
        p._array_form = perm
        p._size = len(perm)
        return p

    def copy(self):
        if False:
            while True:
                i = 10
        return self.__class__(self.array_form)

    def __getnewargs__(self):
        if False:
            print('Hello World!')
        return (self.array_form,)

    def _hashable_content(self):
        if False:
            for i in range(10):
                print('nop')
        return tuple(self.array_form)

    @property
    def array_form(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a copy of the attribute _array_form\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([[2, 0], [3, 1]])\n        >>> p.array_form\n        [2, 3, 0, 1]\n        >>> Permutation([[2, 0, 3, 1]]).array_form\n        [3, 2, 0, 1]\n        >>> Permutation([2, 0, 3, 1]).array_form\n        [2, 0, 3, 1]\n        >>> Permutation([[1, 2], [4, 5]]).array_form\n        [0, 2, 1, 3, 5, 4]\n        '
        return self._array_form[:]

    def list(self, size=None):
        if False:
            return 10
        'Return the permutation as an explicit list, possibly\n        trimming unmoved elements if size is less than the maximum\n        element in the permutation; if this is desired, setting\n        ``size=-1`` will guarantee such trimming.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation(2, 3)(4, 5)\n        >>> p.list()\n        [0, 1, 3, 2, 5, 4]\n        >>> p.list(10)\n        [0, 1, 3, 2, 5, 4, 6, 7, 8, 9]\n\n        Passing a length too small will trim trailing, unchanged elements\n        in the permutation:\n\n        >>> Permutation(2, 4)(1, 2, 4).list(-1)\n        [0, 2, 1]\n        >>> Permutation(3).list(-1)\n        []\n        '
        if not self and size is None:
            raise ValueError('must give size for empty Cycle')
        rv = self.array_form
        if size is not None:
            if size > self.size:
                rv.extend(list(range(self.size, size)))
            else:
                i = self.size - 1
                while rv:
                    if rv[-1] != i:
                        break
                    rv.pop()
                    i -= 1
        return rv

    @property
    def cyclic_form(self):
        if False:
            print('Hello World!')
        '\n        This is used to convert to the cyclic notation\n        from the canonical notation. Singletons are omitted.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([0, 3, 1, 2])\n        >>> p.cyclic_form\n        [[1, 3, 2]]\n        >>> Permutation([1, 0, 2, 4, 3, 5]).cyclic_form\n        [[0, 1], [3, 4]]\n\n        See Also\n        ========\n\n        array_form, full_cyclic_form\n        '
        if self._cyclic_form is not None:
            return list(self._cyclic_form)
        array_form = self.array_form
        unchecked = [True] * len(array_form)
        cyclic_form = []
        for i in range(len(array_form)):
            if unchecked[i]:
                cycle = []
                cycle.append(i)
                unchecked[i] = False
                j = i
                while unchecked[array_form[j]]:
                    j = array_form[j]
                    cycle.append(j)
                    unchecked[j] = False
                if len(cycle) > 1:
                    cyclic_form.append(cycle)
                    assert cycle == list(minlex(cycle))
        cyclic_form.sort()
        self._cyclic_form = cyclic_form[:]
        return cyclic_form

    @property
    def full_cyclic_form(self):
        if False:
            return 10
        'Return permutation in cyclic form including singletons.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> Permutation([0, 2, 1]).full_cyclic_form\n        [[0], [1, 2]]\n        '
        need = set(range(self.size)) - set(flatten(self.cyclic_form))
        rv = self.cyclic_form + [[i] for i in need]
        rv.sort()
        return rv

    @property
    def size(self):
        if False:
            while True:
                i = 10
        '\n        Returns the number of elements in the permutation.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> Permutation([[3, 2], [0, 1]]).size\n        4\n\n        See Also\n        ========\n\n        cardinality, length, order, rank\n        '
        return self._size

    def support(self):
        if False:
            i = 10
            return i + 15
        'Return the elements in permutation, P, for which P[i] != i.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([[3, 2], [0, 1], [4]])\n        >>> p.array_form\n        [1, 0, 3, 2, 4]\n        >>> p.support()\n        [0, 1, 2, 3]\n        '
        a = self.array_form
        return [i for (i, e) in enumerate(a) if a[i] != i]

    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Return permutation that is other higher in rank than self.\n\n        The rank is the lexicographical rank, with the identity permutation\n        having rank of 0.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> I = Permutation([0, 1, 2, 3])\n        >>> a = Permutation([2, 1, 3, 0])\n        >>> I + a.rank() == a\n        True\n\n        See Also\n        ========\n\n        __sub__, inversion_vector\n\n        '
        rank = (self.rank() + other) % self.cardinality
        rv = self.unrank_lex(self.size, rank)
        rv._rank = rank
        return rv

    def __sub__(self, other):
        if False:
            return 10
        'Return the permutation that is other lower in rank than self.\n\n        See Also\n        ========\n\n        __add__\n        '
        return self.__add__(-other)

    @staticmethod
    def rmul(*args):
        if False:
            return 10
        '\n        Return product of Permutations [a, b, c, ...] as the Permutation whose\n        ith value is a(b(c(i))).\n\n        a, b, c, ... can be Permutation objects or tuples.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n\n        >>> a, b = [1, 0, 2], [0, 2, 1]\n        >>> a = Permutation(a); b = Permutation(b)\n        >>> list(Permutation.rmul(a, b))\n        [1, 2, 0]\n        >>> [a(b(i)) for i in range(3)]\n        [1, 2, 0]\n\n        This handles the operands in reverse order compared to the ``*`` operator:\n\n        >>> a = Permutation(a); b = Permutation(b)\n        >>> list(a*b)\n        [2, 0, 1]\n        >>> [b(a(i)) for i in range(3)]\n        [2, 0, 1]\n\n        Notes\n        =====\n\n        All items in the sequence will be parsed by Permutation as\n        necessary as long as the first item is a Permutation:\n\n        >>> Permutation.rmul(a, [0, 2, 1]) == Permutation.rmul(a, b)\n        True\n\n        The reverse order of arguments will raise a TypeError.\n\n        '
        rv = args[0]
        for i in range(1, len(args)):
            rv = args[i] * rv
        return rv

    @classmethod
    def rmul_with_af(cls, *args):
        if False:
            print('Hello World!')
        '\n        same as rmul, but the elements of args are Permutation objects\n        which have _array_form\n        '
        a = [x._array_form for x in args]
        rv = cls._af_new(_af_rmuln(*a))
        return rv

    def mul_inv(self, other):
        if False:
            print('Hello World!')
        '\n        other*~self, self and other have _array_form\n        '
        a = _af_invert(self._array_form)
        b = other._array_form
        return self._af_new(_af_rmul(a, b))

    def __rmul__(self, other):
        if False:
            i = 10
            return i + 15
        'This is needed to coerce other to Permutation in rmul.'
        cls = type(self)
        return cls(other) * self

    def __mul__(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Return the product a*b as a Permutation; the ith value is b(a(i)).\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.permutations import _af_rmul, Permutation\n\n        >>> a, b = [1, 0, 2], [0, 2, 1]\n        >>> a = Permutation(a); b = Permutation(b)\n        >>> list(a*b)\n        [2, 0, 1]\n        >>> [b(a(i)) for i in range(3)]\n        [2, 0, 1]\n\n        This handles operands in reverse order compared to _af_rmul and rmul:\n\n        >>> al = list(a); bl = list(b)\n        >>> _af_rmul(al, bl)\n        [1, 2, 0]\n        >>> [al[bl[i]] for i in range(3)]\n        [1, 2, 0]\n\n        It is acceptable for the arrays to have different lengths; the shorter\n        one will be padded to match the longer one:\n\n        >>> from sympy import init_printing\n        >>> init_printing(perm_cyclic=False, pretty_print=False)\n        >>> b*Permutation([1, 0])\n        Permutation([1, 2, 0])\n        >>> Permutation([1, 0])*b\n        Permutation([2, 0, 1])\n\n        It is also acceptable to allow coercion to handle conversion of a\n        single list to the left of a Permutation:\n\n        >>> [0, 1]*a # no change: 2-element identity\n        Permutation([1, 0, 2])\n        >>> [[0, 1]]*a # exchange first two elements\n        Permutation([0, 1, 2])\n\n        You cannot use more than 1 cycle notation in a product of cycles\n        since coercion can only handle one argument to the left. To handle\n        multiple cycles it is convenient to use Cycle instead of Permutation:\n\n        >>> [[1, 2]]*[[2, 3]]*Permutation([]) # doctest: +SKIP\n        >>> from sympy.combinatorics.permutations import Cycle\n        >>> Cycle(1, 2)(2, 3)\n        (1 3 2)\n\n        '
        from sympy.combinatorics.perm_groups import PermutationGroup, Coset
        if isinstance(other, PermutationGroup):
            return Coset(self, other, dir='-')
        a = self.array_form
        b = other.array_form
        if not b:
            perm = a
        else:
            b.extend(list(range(len(b), len(a))))
            perm = [b[i] for i in a] + b[len(a):]
        return self._af_new(perm)

    def commutes_with(self, other):
        if False:
            while True:
                i = 10
        '\n        Checks if the elements are commuting.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> a = Permutation([1, 4, 3, 0, 2, 5])\n        >>> b = Permutation([0, 1, 2, 3, 4, 5])\n        >>> a.commutes_with(b)\n        True\n        >>> b = Permutation([2, 3, 5, 4, 1, 0])\n        >>> a.commutes_with(b)\n        False\n        '
        a = self.array_form
        b = other.array_form
        return _af_commutes_with(a, b)

    def __pow__(self, n):
        if False:
            i = 10
            return i + 15
        '\n        Routine for finding powers of a permutation.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> from sympy import init_printing\n        >>> init_printing(perm_cyclic=False, pretty_print=False)\n        >>> p = Permutation([2, 0, 3, 1])\n        >>> p.order()\n        4\n        >>> p**4\n        Permutation([0, 1, 2, 3])\n        '
        if isinstance(n, Permutation):
            raise NotImplementedError('p**p is not defined; do you mean p^p (conjugate)?')
        n = int(n)
        return self._af_new(_af_pow(self.array_form, n))

    def __rxor__(self, i):
        if False:
            for i in range(10):
                print('nop')
        'Return self(i) when ``i`` is an int.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation(1, 2, 9)\n        >>> 2^p == p(2) == 9\n        True\n        '
        if int_valued(i):
            return self(i)
        else:
            raise NotImplementedError('i^p = p(i) when i is an integer, not %s.' % i)

    def __xor__(self, h):
        if False:
            return 10
        'Return the conjugate permutation ``~h*self*h` `.\n\n        Explanation\n        ===========\n\n        If ``a`` and ``b`` are conjugates, ``a = h*b*~h`` and\n        ``b = ~h*a*h`` and both have the same cycle structure.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation(1, 2, 9)\n        >>> q = Permutation(6, 9, 8)\n        >>> p*q != q*p\n        True\n\n        Calculate and check properties of the conjugate:\n\n        >>> c = p^q\n        >>> c == ~q*p*q and p == q*c*~q\n        True\n\n        The expression q^p^r is equivalent to q^(p*r):\n\n        >>> r = Permutation(9)(4, 6, 8)\n        >>> q^p^r == q^(p*r)\n        True\n\n        If the term to the left of the conjugate operator, i, is an integer\n        then this is interpreted as selecting the ith element from the\n        permutation to the right:\n\n        >>> all(i^p == p(i) for i in range(p.size))\n        True\n\n        Note that the * operator as higher precedence than the ^ operator:\n\n        >>> q^r*p^r == q^(r*p)^r == Permutation(9)(1, 6, 4)\n        True\n\n        Notes\n        =====\n\n        In Python the precedence rule is p^q^r = (p^q)^r which differs\n        in general from p^(q^r)\n\n        >>> q^p^r\n        (9)(1 4 8)\n        >>> q^(p^r)\n        (9)(1 8 6)\n\n        For a given r and p, both of the following are conjugates of p:\n        ~r*p*r and r*p*~r. But these are not necessarily the same:\n\n        >>> ~r*p*r == r*p*~r\n        True\n\n        >>> p = Permutation(1, 2, 9)(5, 6)\n        >>> ~r*p*r == r*p*~r\n        False\n\n        The conjugate ~r*p*r was chosen so that ``p^q^r`` would be equivalent\n        to ``p^(q*r)`` rather than ``p^(r*q)``. To obtain r*p*~r, pass ~r to\n        this method:\n\n        >>> p^~r == r*p*~r\n        True\n        '
        if self.size != h.size:
            raise ValueError('The permutations must be of equal size.')
        a = [None] * self.size
        h = h._array_form
        p = self._array_form
        for i in range(self.size):
            a[h[i]] = h[p[i]]
        return self._af_new(a)

    def transpositions(self):
        if False:
            i = 10
            return i + 15
        "\n        Return the permutation decomposed into a list of transpositions.\n\n        Explanation\n        ===========\n\n        It is always possible to express a permutation as the product of\n        transpositions, see [1]\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([[1, 2, 3], [0, 4, 5, 6, 7]])\n        >>> t = p.transpositions()\n        >>> t\n        [(0, 7), (0, 6), (0, 5), (0, 4), (1, 3), (1, 2)]\n        >>> print(''.join(str(c) for c in t))\n        (0, 7)(0, 6)(0, 5)(0, 4)(1, 3)(1, 2)\n        >>> Permutation.rmul(*[Permutation([ti], size=p.size) for ti in t]) == p\n        True\n\n        References\n        ==========\n\n        .. [1] https://en.wikipedia.org/wiki/Transposition_%28mathematics%29#Properties\n\n        "
        a = self.cyclic_form
        res = []
        for x in a:
            nx = len(x)
            if nx == 2:
                res.append(tuple(x))
            elif nx > 2:
                first = x[0]
                for y in x[nx - 1:0:-1]:
                    res.append((first, y))
        return res

    @classmethod
    def from_sequence(self, i, key=None):
        if False:
            print('Hello World!')
        'Return the permutation needed to obtain ``i`` from the sorted\n        elements of ``i``. If custom sorting is desired, a key can be given.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n\n        >>> Permutation.from_sequence(\'SymPy\')\n        (4)(0 1 3)\n        >>> _(sorted("SymPy"))\n        [\'S\', \'y\', \'m\', \'P\', \'y\']\n        >>> Permutation.from_sequence(\'SymPy\', key=lambda x: x.lower())\n        (4)(0 2)(1 3)\n        '
        ic = list(zip(i, list(range(len(i)))))
        if key:
            ic.sort(key=lambda x: key(x[0]))
        else:
            ic.sort()
        return ~Permutation([i[1] for i in ic])

    def __invert__(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the inverse of the permutation.\n\n        A permutation multiplied by its inverse is the identity permutation.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> from sympy import init_printing\n        >>> init_printing(perm_cyclic=False, pretty_print=False)\n        >>> p = Permutation([[2, 0], [3, 1]])\n        >>> ~p\n        Permutation([2, 3, 0, 1])\n        >>> _ == p**-1\n        True\n        >>> p*~p == ~p*p == Permutation([0, 1, 2, 3])\n        True\n        '
        return self._af_new(_af_invert(self._array_form))

    def __iter__(self):
        if False:
            print('Hello World!')
        'Yield elements from array form.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> list(Permutation(range(3)))\n        [0, 1, 2]\n        '
        yield from self.array_form

    def __repr__(self):
        if False:
            while True:
                i = 10
        return srepr(self)

    def __call__(self, *i):
        if False:
            print('Hello World!')
        '\n        Allows applying a permutation instance as a bijective function.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([[2, 0], [3, 1]])\n        >>> p.array_form\n        [2, 3, 0, 1]\n        >>> [p(i) for i in range(4)]\n        [2, 3, 0, 1]\n\n        If an array is given then the permutation selects the items\n        from the array (i.e. the permutation is applied to the array):\n\n        >>> from sympy.abc import x\n        >>> p([x, 1, 0, x**2])\n        [0, x**2, x, 1]\n        '
        if len(i) == 1:
            i = i[0]
            if not isinstance(i, Iterable):
                i = as_int(i)
                if i < 0 or i > self.size:
                    raise TypeError('{} should be an integer between 0 and {}'.format(i, self.size - 1))
                return self._array_form[i]
            if len(i) != self.size:
                raise TypeError('{} should have the length {}.'.format(i, self.size))
            return [i[j] for j in self._array_form]
        return self * Permutation(Cycle(*i), size=self.size)

    def atoms(self):
        if False:
            print('Hello World!')
        '\n        Returns all the elements of a permutation\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> Permutation([0, 1, 2, 3, 4, 5]).atoms()\n        {0, 1, 2, 3, 4, 5}\n        >>> Permutation([[0, 1], [2, 3], [4, 5]]).atoms()\n        {0, 1, 2, 3, 4, 5}\n        '
        return set(self.array_form)

    def apply(self, i):
        if False:
            i = 10
            return i + 15
        'Apply the permutation to an expression.\n\n        Parameters\n        ==========\n\n        i : Expr\n            It should be an integer between $0$ and $n-1$ where $n$\n            is the size of the permutation.\n\n            If it is a symbol or a symbolic expression that can\n            have integer values, an ``AppliedPermutation`` object\n            will be returned which can represent an unevaluated\n            function.\n\n        Notes\n        =====\n\n        Any permutation can be defined as a bijective function\n        $\\sigma : \\{ 0, 1, \\dots, n-1 \\} \\rightarrow \\{ 0, 1, \\dots, n-1 \\}$\n        where $n$ denotes the size of the permutation.\n\n        The definition may even be extended for any set with distinctive\n        elements, such that the permutation can even be applied for\n        real numbers or such, however, it is not implemented for now for\n        computational reasons and the integrity with the group theory\n        module.\n\n        This function is similar to the ``__call__`` magic, however,\n        ``__call__`` magic already has some other applications like\n        permuting an array or attaching new cycles, which would\n        not always be mathematically consistent.\n\n        This also guarantees that the return type is a SymPy integer,\n        which guarantees the safety to use assumptions.\n        '
        i = _sympify(i)
        if i.is_integer is False:
            raise NotImplementedError('{} should be an integer.'.format(i))
        n = self.size
        if (i < 0) == True or (i >= n) == True:
            raise NotImplementedError('{} should be an integer between 0 and {}'.format(i, n - 1))
        if i.is_Integer:
            return Integer(self._array_form[i])
        return AppliedPermutation(self, i)

    def next_lex(self):
        if False:
            return 10
        '\n        Returns the next permutation in lexicographical order.\n        If self is the last permutation in lexicographical order\n        it returns None.\n        See [4] section 2.4.\n\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([2, 3, 1, 0])\n        >>> p = Permutation([2, 3, 1, 0]); p.rank()\n        17\n        >>> p = p.next_lex(); p.rank()\n        18\n\n        See Also\n        ========\n\n        rank, unrank_lex\n        '
        perm = self.array_form[:]
        n = len(perm)
        i = n - 2
        while perm[i + 1] < perm[i]:
            i -= 1
        if i == -1:
            return None
        else:
            j = n - 1
            while perm[j] < perm[i]:
                j -= 1
            (perm[j], perm[i]) = (perm[i], perm[j])
            i += 1
            j = n - 1
            while i < j:
                (perm[j], perm[i]) = (perm[i], perm[j])
                i += 1
                j -= 1
        return self._af_new(perm)

    @classmethod
    def unrank_nonlex(self, n, r):
        if False:
            i = 10
            return i + 15
        '\n        This is a linear time unranking algorithm that does not\n        respect lexicographic order [3].\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> from sympy import init_printing\n        >>> init_printing(perm_cyclic=False, pretty_print=False)\n        >>> Permutation.unrank_nonlex(4, 5)\n        Permutation([2, 0, 3, 1])\n        >>> Permutation.unrank_nonlex(4, -1)\n        Permutation([0, 1, 2, 3])\n\n        See Also\n        ========\n\n        next_nonlex, rank_nonlex\n        '

        def _unrank1(n, r, a):
            if False:
                i = 10
                return i + 15
            if n > 0:
                (a[n - 1], a[r % n]) = (a[r % n], a[n - 1])
                _unrank1(n - 1, r // n, a)
        id_perm = list(range(n))
        n = int(n)
        r = r % ifac(n)
        _unrank1(n, r, id_perm)
        return self._af_new(id_perm)

    def rank_nonlex(self, inv_perm=None):
        if False:
            while True:
                i = 10
        '\n        This is a linear time ranking algorithm that does not\n        enforce lexicographic order [3].\n\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([0, 1, 2, 3])\n        >>> p.rank_nonlex()\n        23\n\n        See Also\n        ========\n\n        next_nonlex, unrank_nonlex\n        '

        def _rank1(n, perm, inv_perm):
            if False:
                for i in range(10):
                    print('nop')
            if n == 1:
                return 0
            s = perm[n - 1]
            t = inv_perm[n - 1]
            (perm[n - 1], perm[t]) = (perm[t], s)
            (inv_perm[n - 1], inv_perm[s]) = (inv_perm[s], t)
            return s + n * _rank1(n - 1, perm, inv_perm)
        if inv_perm is None:
            inv_perm = (~self).array_form
        if not inv_perm:
            return 0
        perm = self.array_form[:]
        r = _rank1(len(perm), perm, inv_perm)
        return r

    def next_nonlex(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the next permutation in nonlex order [3].\n        If self is the last permutation in this order it returns None.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> from sympy import init_printing\n        >>> init_printing(perm_cyclic=False, pretty_print=False)\n        >>> p = Permutation([2, 0, 3, 1]); p.rank_nonlex()\n        5\n        >>> p = p.next_nonlex(); p\n        Permutation([3, 0, 1, 2])\n        >>> p.rank_nonlex()\n        6\n\n        See Also\n        ========\n\n        rank_nonlex, unrank_nonlex\n        '
        r = self.rank_nonlex()
        if r == ifac(self.size) - 1:
            return None
        return self.unrank_nonlex(self.size, r + 1)

    def rank(self):
        if False:
            print('Hello World!')
        '\n        Returns the lexicographic rank of the permutation.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([0, 1, 2, 3])\n        >>> p.rank()\n        0\n        >>> p = Permutation([3, 2, 1, 0])\n        >>> p.rank()\n        23\n\n        See Also\n        ========\n\n        next_lex, unrank_lex, cardinality, length, order, size\n        '
        if self._rank is not None:
            return self._rank
        rank = 0
        rho = self.array_form[:]
        n = self.size - 1
        size = n + 1
        psize = int(ifac(n))
        for j in range(size - 1):
            rank += rho[j] * psize
            for i in range(j + 1, size):
                if rho[i] > rho[j]:
                    rho[i] -= 1
            psize //= n
            n -= 1
        self._rank = rank
        return rank

    @property
    def cardinality(self):
        if False:
            while True:
                i = 10
        '\n        Returns the number of all possible permutations.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([0, 1, 2, 3])\n        >>> p.cardinality\n        24\n\n        See Also\n        ========\n\n        length, order, rank, size\n        '
        return int(ifac(self.size))

    def parity(self):
        if False:
            print('Hello World!')
        '\n        Computes the parity of a permutation.\n\n        Explanation\n        ===========\n\n        The parity of a permutation reflects the parity of the\n        number of inversions in the permutation, i.e., the\n        number of pairs of x and y such that ``x > y`` but ``p[x] < p[y]``.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([0, 1, 2, 3])\n        >>> p.parity()\n        0\n        >>> p = Permutation([3, 2, 0, 1])\n        >>> p.parity()\n        1\n\n        See Also\n        ========\n\n        _af_parity\n        '
        if self._cyclic_form is not None:
            return (self.size - self.cycles) % 2
        return _af_parity(self.array_form)

    @property
    def is_even(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks if a permutation is even.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([0, 1, 2, 3])\n        >>> p.is_even\n        True\n        >>> p = Permutation([3, 2, 1, 0])\n        >>> p.is_even\n        True\n\n        See Also\n        ========\n\n        is_odd\n        '
        return not self.is_odd

    @property
    def is_odd(self):
        if False:
            print('Hello World!')
        '\n        Checks if a permutation is odd.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([0, 1, 2, 3])\n        >>> p.is_odd\n        False\n        >>> p = Permutation([3, 2, 0, 1])\n        >>> p.is_odd\n        True\n\n        See Also\n        ========\n\n        is_even\n        '
        return bool(self.parity() % 2)

    @property
    def is_Singleton(self):
        if False:
            i = 10
            return i + 15
        '\n        Checks to see if the permutation contains only one number and is\n        thus the only possible permutation of this set of numbers\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> Permutation([0]).is_Singleton\n        True\n        >>> Permutation([0, 1]).is_Singleton\n        False\n\n        See Also\n        ========\n\n        is_Empty\n        '
        return self.size == 1

    @property
    def is_Empty(self):
        if False:
            while True:
                i = 10
        '\n        Checks to see if the permutation is a set with zero elements\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> Permutation([]).is_Empty\n        True\n        >>> Permutation([0]).is_Empty\n        False\n\n        See Also\n        ========\n\n        is_Singleton\n        '
        return self.size == 0

    @property
    def is_identity(self):
        if False:
            i = 10
            return i + 15
        return self.is_Identity

    @property
    def is_Identity(self):
        if False:
            print('Hello World!')
        '\n        Returns True if the Permutation is an identity permutation.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([])\n        >>> p.is_Identity\n        True\n        >>> p = Permutation([[0], [1], [2]])\n        >>> p.is_Identity\n        True\n        >>> p = Permutation([0, 1, 2])\n        >>> p.is_Identity\n        True\n        >>> p = Permutation([0, 2, 1])\n        >>> p.is_Identity\n        False\n\n        See Also\n        ========\n\n        order\n        '
        af = self.array_form
        return not af or all((i == af[i] for i in range(self.size)))

    def ascents(self):
        if False:
            while True:
                i = 10
        '\n        Returns the positions of ascents in a permutation, ie, the location\n        where p[i] < p[i+1]\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([4, 0, 1, 3, 2])\n        >>> p.ascents()\n        [1, 2]\n\n        See Also\n        ========\n\n        descents, inversions, min, max\n        '
        a = self.array_form
        pos = [i for i in range(len(a) - 1) if a[i] < a[i + 1]]
        return pos

    def descents(self):
        if False:
            print('Hello World!')
        '\n        Returns the positions of descents in a permutation, ie, the location\n        where p[i] > p[i+1]\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([4, 0, 1, 3, 2])\n        >>> p.descents()\n        [0, 3]\n\n        See Also\n        ========\n\n        ascents, inversions, min, max\n        '
        a = self.array_form
        pos = [i for i in range(len(a) - 1) if a[i] > a[i + 1]]
        return pos

    def max(self):
        if False:
            print('Hello World!')
        '\n        The maximum element moved by the permutation.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([1, 0, 2, 3, 4])\n        >>> p.max()\n        1\n\n        See Also\n        ========\n\n        min, descents, ascents, inversions\n        '
        max = 0
        a = self.array_form
        for i in range(len(a)):
            if a[i] != i and a[i] > max:
                max = a[i]
        return max

    def min(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The minimum element moved by the permutation.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([0, 1, 4, 3, 2])\n        >>> p.min()\n        2\n\n        See Also\n        ========\n\n        max, descents, ascents, inversions\n        '
        a = self.array_form
        min = len(a)
        for i in range(len(a)):
            if a[i] != i and a[i] < min:
                min = a[i]
        return min

    def inversions(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Computes the number of inversions of a permutation.\n\n        Explanation\n        ===========\n\n        An inversion is where i > j but p[i] < p[j].\n\n        For small length of p, it iterates over all i and j\n        values and calculates the number of inversions.\n        For large length of p, it uses a variation of merge\n        sort to calculate the number of inversions.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([0, 1, 2, 3, 4, 5])\n        >>> p.inversions()\n        0\n        >>> Permutation([3, 2, 1, 0]).inversions()\n        6\n\n        See Also\n        ========\n\n        descents, ascents, min, max\n\n        References\n        ==========\n\n        .. [1] https://www.cp.eng.chula.ac.th/~prabhas//teaching/algo/algo2008/count-inv.htm\n\n        '
        inversions = 0
        a = self.array_form
        n = len(a)
        if n < 130:
            for i in range(n - 1):
                b = a[i]
                for c in a[i + 1:]:
                    if b > c:
                        inversions += 1
        else:
            k = 1
            right = 0
            arr = a[:]
            temp = a[:]
            while k < n:
                i = 0
                while i + k < n:
                    right = i + k * 2 - 1
                    if right >= n:
                        right = n - 1
                    inversions += _merge(arr, temp, i, i + k, right)
                    i = i + k * 2
                k = k * 2
        return inversions

    def commutator(self, x):
        if False:
            while True:
                i = 10
        'Return the commutator of ``self`` and ``x``: ``~x*~self*x*self``\n\n        If f and g are part of a group, G, then the commutator of f and g\n        is the group identity iff f and g commute, i.e. fg == gf.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> from sympy import init_printing\n        >>> init_printing(perm_cyclic=False, pretty_print=False)\n        >>> p = Permutation([0, 2, 3, 1])\n        >>> x = Permutation([2, 0, 3, 1])\n        >>> c = p.commutator(x); c\n        Permutation([2, 1, 3, 0])\n        >>> c == ~x*~p*x*p\n        True\n\n        >>> I = Permutation(3)\n        >>> p = [I + i for i in range(6)]\n        >>> for i in range(len(p)):\n        ...     for j in range(len(p)):\n        ...         c = p[i].commutator(p[j])\n        ...         if p[i]*p[j] == p[j]*p[i]:\n        ...             assert c == I\n        ...         else:\n        ...             assert c != I\n        ...\n\n        References\n        ==========\n\n        .. [1] https://en.wikipedia.org/wiki/Commutator\n        '
        a = self.array_form
        b = x.array_form
        n = len(a)
        if len(b) != n:
            raise ValueError('The permutations must be of equal size.')
        inva = [None] * n
        for i in range(n):
            inva[a[i]] = i
        invb = [None] * n
        for i in range(n):
            invb[b[i]] = i
        return self._af_new([a[b[inva[i]]] for i in invb])

    def signature(self):
        if False:
            i = 10
            return i + 15
        '\n        Gives the signature of the permutation needed to place the\n        elements of the permutation in canonical order.\n\n        The signature is calculated as (-1)^<number of inversions>\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([0, 1, 2])\n        >>> p.inversions()\n        0\n        >>> p.signature()\n        1\n        >>> q = Permutation([0,2,1])\n        >>> q.inversions()\n        1\n        >>> q.signature()\n        -1\n\n        See Also\n        ========\n\n        inversions\n        '
        if self.is_even:
            return 1
        return -1

    def order(self):
        if False:
            return 10
        '\n        Computes the order of a permutation.\n\n        When the permutation is raised to the power of its\n        order it equals the identity permutation.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> from sympy import init_printing\n        >>> init_printing(perm_cyclic=False, pretty_print=False)\n        >>> p = Permutation([3, 1, 5, 2, 4, 0])\n        >>> p.order()\n        4\n        >>> (p**(p.order()))\n        Permutation([], size=6)\n\n        See Also\n        ========\n\n        identity, cardinality, length, rank, size\n        '
        return reduce(lcm, [len(cycle) for cycle in self.cyclic_form], 1)

    def length(self):
        if False:
            while True:
                i = 10
        '\n        Returns the number of integers moved by a permutation.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> Permutation([0, 3, 2, 1]).length()\n        2\n        >>> Permutation([[0, 1], [2, 3]]).length()\n        4\n\n        See Also\n        ========\n\n        min, max, support, cardinality, order, rank, size\n        '
        return len(self.support())

    @property
    def cycle_structure(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the cycle structure of the permutation as a dictionary\n        indicating the multiplicity of each cycle length.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> Permutation(3).cycle_structure\n        {1: 4}\n        >>> Permutation(0, 4, 3)(1, 2)(5, 6).cycle_structure\n        {2: 2, 3: 1}\n        '
        if self._cycle_structure:
            rv = self._cycle_structure
        else:
            rv = defaultdict(int)
            singletons = self.size
            for c in self.cyclic_form:
                rv[len(c)] += 1
                singletons -= len(c)
            if singletons:
                rv[1] = singletons
            self._cycle_structure = rv
        return dict(rv)

    @property
    def cycles(self):
        if False:
            return 10
        '\n        Returns the number of cycles contained in the permutation\n        (including singletons).\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> Permutation([0, 1, 2]).cycles\n        3\n        >>> Permutation([0, 1, 2]).full_cyclic_form\n        [[0], [1], [2]]\n        >>> Permutation(0, 1)(2, 3).cycles\n        2\n\n        See Also\n        ========\n        sympy.functions.combinatorial.numbers.stirling\n        '
        return len(self.full_cyclic_form)

    def index(self):
        if False:
            print('Hello World!')
        '\n        Returns the index of a permutation.\n\n        The index of a permutation is the sum of all subscripts j such\n        that p[j] is greater than p[j+1].\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([3, 0, 2, 1, 4])\n        >>> p.index()\n        2\n        '
        a = self.array_form
        return sum([j for j in range(len(a) - 1) if a[j] > a[j + 1]])

    def runs(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the runs of a permutation.\n\n        An ascending sequence in a permutation is called a run [5].\n\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([2, 5, 7, 3, 6, 0, 1, 4, 8])\n        >>> p.runs()\n        [[2, 5, 7], [3, 6], [0, 1, 4, 8]]\n        >>> q = Permutation([1,3,2,0])\n        >>> q.runs()\n        [[1, 3], [2], [0]]\n        '
        return runs(self.array_form)

    def inversion_vector(self):
        if False:
            i = 10
            return i + 15
        "Return the inversion vector of the permutation.\n\n        The inversion vector consists of elements whose value\n        indicates the number of elements in the permutation\n        that are lesser than it and lie on its right hand side.\n\n        The inversion vector is the same as the Lehmer encoding of a\n        permutation.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([4, 8, 0, 7, 1, 5, 3, 6, 2])\n        >>> p.inversion_vector()\n        [4, 7, 0, 5, 0, 2, 1, 1]\n        >>> p = Permutation([3, 2, 1, 0])\n        >>> p.inversion_vector()\n        [3, 2, 1]\n\n        The inversion vector increases lexicographically with the rank\n        of the permutation, the -ith element cycling through 0..i.\n\n        >>> p = Permutation(2)\n        >>> while p:\n        ...     print('%s %s %s' % (p, p.inversion_vector(), p.rank()))\n        ...     p = p.next_lex()\n        (2) [0, 0] 0\n        (1 2) [0, 1] 1\n        (2)(0 1) [1, 0] 2\n        (0 1 2) [1, 1] 3\n        (0 2 1) [2, 0] 4\n        (0 2) [2, 1] 5\n\n        See Also\n        ========\n\n        from_inversion_vector\n        "
        self_array_form = self.array_form
        n = len(self_array_form)
        inversion_vector = [0] * (n - 1)
        for i in range(n - 1):
            val = 0
            for j in range(i + 1, n):
                if self_array_form[j] < self_array_form[i]:
                    val += 1
            inversion_vector[i] = val
        return inversion_vector

    def rank_trotterjohnson(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the Trotter Johnson rank, which we get from the minimal\n        change algorithm. See [4] section 2.4.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([0, 1, 2, 3])\n        >>> p.rank_trotterjohnson()\n        0\n        >>> p = Permutation([0, 2, 1, 3])\n        >>> p.rank_trotterjohnson()\n        7\n\n        See Also\n        ========\n\n        unrank_trotterjohnson, next_trotterjohnson\n        '
        if self.array_form == [] or self.is_Identity:
            return 0
        if self.array_form == [1, 0]:
            return 1
        perm = self.array_form
        n = self.size
        rank = 0
        for j in range(1, n):
            k = 1
            i = 0
            while perm[i] != j:
                if perm[i] < j:
                    k += 1
                i += 1
            j1 = j + 1
            if rank % 2 == 0:
                rank = j1 * rank + j1 - k
            else:
                rank = j1 * rank + k - 1
        return rank

    @classmethod
    def unrank_trotterjohnson(cls, size, rank):
        if False:
            while True:
                i = 10
        '\n        Trotter Johnson permutation unranking. See [4] section 2.4.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> from sympy import init_printing\n        >>> init_printing(perm_cyclic=False, pretty_print=False)\n        >>> Permutation.unrank_trotterjohnson(5, 10)\n        Permutation([0, 3, 1, 2, 4])\n\n        See Also\n        ========\n\n        rank_trotterjohnson, next_trotterjohnson\n        '
        perm = [0] * size
        r2 = 0
        n = ifac(size)
        pj = 1
        for j in range(2, size + 1):
            pj *= j
            r1 = rank * pj // n
            k = r1 - j * r2
            if r2 % 2 == 0:
                for i in range(j - 1, j - k - 1, -1):
                    perm[i] = perm[i - 1]
                perm[j - k - 1] = j - 1
            else:
                for i in range(j - 1, k, -1):
                    perm[i] = perm[i - 1]
                perm[k] = j - 1
            r2 = r1
        return cls._af_new(perm)

    def next_trotterjohnson(self):
        if False:
            print('Hello World!')
        '\n        Returns the next permutation in Trotter-Johnson order.\n        If self is the last permutation it returns None.\n        See [4] section 2.4. If it is desired to generate all such\n        permutations, they can be generated in order more quickly\n        with the ``generate_bell`` function.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> from sympy import init_printing\n        >>> init_printing(perm_cyclic=False, pretty_print=False)\n        >>> p = Permutation([3, 0, 2, 1])\n        >>> p.rank_trotterjohnson()\n        4\n        >>> p = p.next_trotterjohnson(); p\n        Permutation([0, 3, 2, 1])\n        >>> p.rank_trotterjohnson()\n        5\n\n        See Also\n        ========\n\n        rank_trotterjohnson, unrank_trotterjohnson, sympy.utilities.iterables.generate_bell\n        '
        pi = self.array_form[:]
        n = len(pi)
        st = 0
        rho = pi[:]
        done = False
        m = n - 1
        while m > 0 and (not done):
            d = rho.index(m)
            for i in range(d, m):
                rho[i] = rho[i + 1]
            par = _af_parity(rho[:m])
            if par == 1:
                if d == m:
                    m -= 1
                else:
                    (pi[st + d], pi[st + d + 1]) = (pi[st + d + 1], pi[st + d])
                    done = True
            elif d == 0:
                m -= 1
                st += 1
            else:
                (pi[st + d], pi[st + d - 1]) = (pi[st + d - 1], pi[st + d])
                done = True
        if m == 0:
            return None
        return self._af_new(pi)

    def get_precedence_matrix(self):
        if False:
            while True:
                i = 10
        '\n        Gets the precedence matrix. This is used for computing the\n        distance between two permutations.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> from sympy import init_printing\n        >>> init_printing(perm_cyclic=False, pretty_print=False)\n        >>> p = Permutation.josephus(3, 6, 1)\n        >>> p\n        Permutation([2, 5, 3, 1, 4, 0])\n        >>> p.get_precedence_matrix()\n        Matrix([\n        [0, 0, 0, 0, 0, 0],\n        [1, 0, 0, 0, 1, 0],\n        [1, 1, 0, 1, 1, 1],\n        [1, 1, 0, 0, 1, 0],\n        [1, 0, 0, 0, 0, 0],\n        [1, 1, 0, 1, 1, 0]])\n\n        See Also\n        ========\n\n        get_precedence_distance, get_adjacency_matrix, get_adjacency_distance\n        '
        m = zeros(self.size)
        perm = self.array_form
        for i in range(m.rows):
            for j in range(i + 1, m.cols):
                m[perm[i], perm[j]] = 1
        return m

    def get_precedence_distance(self, other):
        if False:
            return 10
        "\n        Computes the precedence distance between two permutations.\n\n        Explanation\n        ===========\n\n        Suppose p and p' represent n jobs. The precedence metric\n        counts the number of times a job j is preceded by job i\n        in both p and p'. This metric is commutative.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([2, 0, 4, 3, 1])\n        >>> q = Permutation([3, 1, 2, 4, 0])\n        >>> p.get_precedence_distance(q)\n        7\n        >>> q.get_precedence_distance(p)\n        7\n\n        See Also\n        ========\n\n        get_precedence_matrix, get_adjacency_matrix, get_adjacency_distance\n        "
        if self.size != other.size:
            raise ValueError('The permutations must be of equal size.')
        self_prec_mat = self.get_precedence_matrix()
        other_prec_mat = other.get_precedence_matrix()
        n_prec = 0
        for i in range(self.size):
            for j in range(self.size):
                if i == j:
                    continue
                if self_prec_mat[i, j] * other_prec_mat[i, j] == 1:
                    n_prec += 1
        d = self.size * (self.size - 1) // 2 - n_prec
        return d

    def get_adjacency_matrix(self):
        if False:
            i = 10
            return i + 15
        '\n        Computes the adjacency matrix of a permutation.\n\n        Explanation\n        ===========\n\n        If job i is adjacent to job j in a permutation p\n        then we set m[i, j] = 1 where m is the adjacency\n        matrix of p.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation.josephus(3, 6, 1)\n        >>> p.get_adjacency_matrix()\n        Matrix([\n        [0, 0, 0, 0, 0, 0],\n        [0, 0, 0, 0, 1, 0],\n        [0, 0, 0, 0, 0, 1],\n        [0, 1, 0, 0, 0, 0],\n        [1, 0, 0, 0, 0, 0],\n        [0, 0, 0, 1, 0, 0]])\n        >>> q = Permutation([0, 1, 2, 3])\n        >>> q.get_adjacency_matrix()\n        Matrix([\n        [0, 1, 0, 0],\n        [0, 0, 1, 0],\n        [0, 0, 0, 1],\n        [0, 0, 0, 0]])\n\n        See Also\n        ========\n\n        get_precedence_matrix, get_precedence_distance, get_adjacency_distance\n        '
        m = zeros(self.size)
        perm = self.array_form
        for i in range(self.size - 1):
            m[perm[i], perm[i + 1]] = 1
        return m

    def get_adjacency_distance(self, other):
        if False:
            print('Hello World!')
        "\n        Computes the adjacency distance between two permutations.\n\n        Explanation\n        ===========\n\n        This metric counts the number of times a pair i,j of jobs is\n        adjacent in both p and p'. If n_adj is this quantity then\n        the adjacency distance is n - n_adj - 1 [1]\n\n        [1] Reeves, Colin R. Landscapes, Operators and Heuristic search, Annals\n        of Operational Research, 86, pp 473-490. (1999)\n\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([0, 3, 1, 2, 4])\n        >>> q = Permutation.josephus(4, 5, 2)\n        >>> p.get_adjacency_distance(q)\n        3\n        >>> r = Permutation([0, 2, 1, 4, 3])\n        >>> p.get_adjacency_distance(r)\n        4\n\n        See Also\n        ========\n\n        get_precedence_matrix, get_precedence_distance, get_adjacency_matrix\n        "
        if self.size != other.size:
            raise ValueError('The permutations must be of the same size.')
        self_adj_mat = self.get_adjacency_matrix()
        other_adj_mat = other.get_adjacency_matrix()
        n_adj = 0
        for i in range(self.size):
            for j in range(self.size):
                if i == j:
                    continue
                if self_adj_mat[i, j] * other_adj_mat[i, j] == 1:
                    n_adj += 1
        d = self.size - n_adj - 1
        return d

    def get_positional_distance(self, other):
        if False:
            return 10
        '\n        Computes the positional distance between two permutations.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> p = Permutation([0, 3, 1, 2, 4])\n        >>> q = Permutation.josephus(4, 5, 2)\n        >>> r = Permutation([3, 1, 4, 0, 2])\n        >>> p.get_positional_distance(q)\n        12\n        >>> p.get_positional_distance(r)\n        12\n\n        See Also\n        ========\n\n        get_precedence_distance, get_adjacency_distance\n        '
        a = self.array_form
        b = other.array_form
        if len(a) != len(b):
            raise ValueError('The permutations must be of the same size.')
        return sum([abs(a[i] - b[i]) for i in range(len(a))])

    @classmethod
    def josephus(cls, m, n, s=1):
        if False:
            i = 10
            return i + 15
        'Return as a permutation the shuffling of range(n) using the Josephus\n        scheme in which every m-th item is selected until all have been chosen.\n        The returned permutation has elements listed by the order in which they\n        were selected.\n\n        The parameter ``s`` stops the selection process when there are ``s``\n        items remaining and these are selected by continuing the selection,\n        counting by 1 rather than by ``m``.\n\n        Consider selecting every 3rd item from 6 until only 2 remain::\n\n            choices    chosen\n            ========   ======\n              012345\n              01 345   2\n              01 34    25\n              01  4    253\n              0   4    2531\n              0        25314\n                       253140\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> Permutation.josephus(3, 6, 2).array_form\n        [2, 5, 3, 1, 4, 0]\n\n        References\n        ==========\n\n        .. [1] https://en.wikipedia.org/wiki/Flavius_Josephus\n        .. [2] https://en.wikipedia.org/wiki/Josephus_problem\n        .. [3] https://web.archive.org/web/20171008094331/http://www.wou.edu/~burtonl/josephus.html\n\n        '
        from collections import deque
        m -= 1
        Q = deque(list(range(n)))
        perm = []
        while len(Q) > max(s, 1):
            for dp in range(m):
                Q.append(Q.popleft())
            perm.append(Q.popleft())
        perm.extend(list(Q))
        return cls(perm)

    @classmethod
    def from_inversion_vector(cls, inversion):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculates the permutation from the inversion vector.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> from sympy import init_printing\n        >>> init_printing(perm_cyclic=False, pretty_print=False)\n        >>> Permutation.from_inversion_vector([3, 2, 1, 0, 0])\n        Permutation([3, 2, 1, 0, 4, 5])\n\n        '
        size = len(inversion)
        N = list(range(size + 1))
        perm = []
        try:
            for k in range(size):
                val = N[inversion[k]]
                perm.append(val)
                N.remove(val)
        except IndexError:
            raise ValueError('The inversion vector is not valid.')
        perm.extend(N)
        return cls._af_new(perm)

    @classmethod
    def random(cls, n):
        if False:
            print('Hello World!')
        '\n        Generates a random permutation of length ``n``.\n\n        Uses the underlying Python pseudo-random number generator.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> Permutation.random(2) in (Permutation([1, 0]), Permutation([0, 1]))\n        True\n\n        '
        perm_array = list(range(n))
        random.shuffle(perm_array)
        return cls._af_new(perm_array)

    @classmethod
    def unrank_lex(cls, size, rank):
        if False:
            while True:
                i = 10
        '\n        Lexicographic permutation unranking.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n        >>> from sympy import init_printing\n        >>> init_printing(perm_cyclic=False, pretty_print=False)\n        >>> a = Permutation.unrank_lex(5, 10)\n        >>> a.rank()\n        10\n        >>> a\n        Permutation([0, 2, 4, 1, 3])\n\n        See Also\n        ========\n\n        rank, next_lex\n        '
        perm_array = [0] * size
        psize = 1
        for i in range(size):
            new_psize = psize * (i + 1)
            d = rank % new_psize // psize
            rank -= d * psize
            perm_array[size - i - 1] = d
            for j in range(size - i, size):
                if perm_array[j] > d - 1:
                    perm_array[j] += 1
            psize = new_psize
        return cls._af_new(perm_array)

    def resize(self, n):
        if False:
            print('Hello World!')
        'Resize the permutation to the new size ``n``.\n\n        Parameters\n        ==========\n\n        n : int\n            The new size of the permutation.\n\n        Raises\n        ======\n\n        ValueError\n            If the permutation cannot be resized to the given size.\n            This may only happen when resized to a smaller size than\n            the original.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics import Permutation\n\n        Increasing the size of a permutation:\n\n        >>> p = Permutation(0, 1, 2)\n        >>> p = p.resize(5)\n        >>> p\n        (4)(0 1 2)\n\n        Decreasing the size of the permutation:\n\n        >>> p = p.resize(4)\n        >>> p\n        (3)(0 1 2)\n\n        If resizing to the specific size breaks the cycles:\n\n        >>> p.resize(2)\n        Traceback (most recent call last):\n        ...\n        ValueError: The permutation cannot be resized to 2 because the\n        cycle (0, 1, 2) may break.\n        '
        aform = self.array_form
        l = len(aform)
        if n > l:
            aform += list(range(l, n))
            return Permutation._af_new(aform)
        elif n < l:
            cyclic_form = self.full_cyclic_form
            new_cyclic_form = []
            for cycle in cyclic_form:
                cycle_min = min(cycle)
                cycle_max = max(cycle)
                if cycle_min <= n - 1:
                    if cycle_max > n - 1:
                        raise ValueError('The permutation cannot be resized to {} because the cycle {} may break.'.format(n, tuple(cycle)))
                    new_cyclic_form.append(cycle)
            return Permutation(new_cyclic_form)
        return self
    print_cyclic = None

def _merge(arr, temp, left, mid, right):
    if False:
        i = 10
        return i + 15
    '\n    Merges two sorted arrays and calculates the inversion count.\n\n    Helper function for calculating inversions. This method is\n    for internal use only.\n    '
    i = k = left
    j = mid
    inv_count = 0
    while i < mid and j <= right:
        if arr[i] < arr[j]:
            temp[k] = arr[i]
            k += 1
            i += 1
        else:
            temp[k] = arr[j]
            k += 1
            j += 1
            inv_count += mid - i
    while i < mid:
        temp[k] = arr[i]
        k += 1
        i += 1
    if j <= right:
        k += right - j + 1
        j += right - j + 1
        arr[left:k + 1] = temp[left:k + 1]
    else:
        arr[left:right + 1] = temp[left:right + 1]
    return inv_count
Perm = Permutation
_af_new = Perm._af_new

class AppliedPermutation(Expr):
    """A permutation applied to a symbolic variable.

    Parameters
    ==========

    perm : Permutation
    x : Expr

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.combinatorics import Permutation

    Creating a symbolic permutation function application:

    >>> x = Symbol('x')
    >>> p = Permutation(0, 1, 2)
    >>> p.apply(x)
    AppliedPermutation((0 1 2), x)
    >>> _.subs(x, 1)
    2
    """

    def __new__(cls, perm, x, evaluate=None):
        if False:
            i = 10
            return i + 15
        if evaluate is None:
            evaluate = global_parameters.evaluate
        perm = _sympify(perm)
        x = _sympify(x)
        if not isinstance(perm, Permutation):
            raise ValueError('{} must be a Permutation instance.'.format(perm))
        if evaluate:
            if x.is_Integer:
                return perm.apply(x)
        obj = super().__new__(cls, perm, x)
        return obj

@dispatch(Permutation, Permutation)
def _eval_is_eq(lhs, rhs):
    if False:
        for i in range(10):
            print('nop')
    if lhs._size != rhs._size:
        return None
    return lhs._array_form == rhs._array_form