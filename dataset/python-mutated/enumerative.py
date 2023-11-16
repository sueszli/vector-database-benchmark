"""
Algorithms and classes to support enumerative combinatorics.

Currently just multiset partitions, but more could be added.

Terminology (following Knuth, algorithm 7.1.2.5M TAOCP)
*multiset* aaabbcccc has a *partition* aaabc | bccc

The submultisets, aaabc and bccc of the partition are called
*parts*, or sometimes *vectors*.  (Knuth notes that multiset
partitions can be thought of as partitions of vectors of integers,
where the ith element of the vector gives the multiplicity of
element i.)

The values a, b and c are *components* of the multiset.  These
correspond to elements of a set, but in a multiset can be present
with a multiplicity greater than 1.

The algorithm deserves some explanation.

Think of the part aaabc from the multiset above.  If we impose an
ordering on the components of the multiset, we can represent a part
with a vector, in which the value of the first element of the vector
corresponds to the multiplicity of the first component in that
part. Thus, aaabc can be represented by the vector [3, 1, 1].  We
can also define an ordering on parts, based on the lexicographic
ordering of the vector (leftmost vector element, i.e., the element
with the smallest component number, is the most significant), so
that [3, 1, 1] > [3, 1, 0] and [3, 1, 1] > [2, 1, 4].  The ordering
on parts can be extended to an ordering on partitions: First, sort
the parts in each partition, left-to-right in decreasing order. Then
partition A is greater than partition B if A's leftmost/greatest
part is greater than B's leftmost part.  If the leftmost parts are
equal, compare the second parts, and so on.

In this ordering, the greatest partition of a given multiset has only
one part.  The least partition is the one in which the components
are spread out, one per part.

The enumeration algorithms in this file yield the partitions of the
argument multiset in decreasing order.  The main data structure is a
stack of parts, corresponding to the current partition.  An
important invariant is that the parts on the stack are themselves in
decreasing order.  This data structure is decremented to find the
next smaller partition.  Most often, decrementing the partition will
only involve adjustments to the smallest parts at the top of the
stack, much as adjacent integers *usually* differ only in their last
few digits.

Knuth's algorithm uses two main operations on parts:

Decrement - change the part so that it is smaller in the
  (vector) lexicographic order, but reduced by the smallest amount possible.
  For example, if the multiset has vector [5,
  3, 1], and the bottom/greatest part is [4, 2, 1], this part would
  decrement to [4, 2, 0], while [4, 0, 0] would decrement to [3, 3,
  1].  A singleton part is never decremented -- [1, 0, 0] is not
  decremented to [0, 3, 1].  Instead, the decrement operator needs
  to fail for this case.  In Knuth's pseudocode, the decrement
  operator is step m5.

Spread unallocated multiplicity - Once a part has been decremented,
  it cannot be the rightmost part in the partition.  There is some
  multiplicity that has not been allocated, and new parts must be
  created above it in the stack to use up this multiplicity.  To
  maintain the invariant that the parts on the stack are in
  decreasing order, these new parts must be less than or equal to
  the decremented part.
  For example, if the multiset is [5, 3, 1], and its most
  significant part has just been decremented to [5, 3, 0], the
  spread operation will add a new part so that the stack becomes
  [[5, 3, 0], [0, 0, 1]].  If the most significant part (for the
  same multiset) has been decremented to [2, 0, 0] the stack becomes
  [[2, 0, 0], [2, 0, 0], [1, 3, 1]].  In the pseudocode, the spread
  operation for one part is step m2.  The complete spread operation
  is a loop of steps m2 and m3.

In order to facilitate the spread operation, Knuth stores, for each
component of each part, not just the multiplicity of that component
in the part, but also the total multiplicity available for this
component in this part or any lesser part above it on the stack.

One added twist is that Knuth does not represent the part vectors as
arrays. Instead, he uses a sparse representation, in which a
component of a part is represented as a component number (c), plus
the multiplicity of the component in that part (v) as well as the
total multiplicity available for that component (u).  This saves
time that would be spent skipping over zeros.

"""

class PartComponent:
    """Internal class used in support of the multiset partitions
    enumerators and the associated visitor functions.

    Represents one component of one part of the current partition.

    A stack of these, plus an auxiliary frame array, f, represents a
    partition of the multiset.

    Knuth's pseudocode makes c, u, and v separate arrays.
    """
    __slots__ = ('c', 'u', 'v')

    def __init__(self):
        if False:
            while True:
                i = 10
        self.c = 0
        self.u = 0
        self.v = 0

    def __repr__(self):
        if False:
            print('Hello World!')
        'for debug/algorithm animation purposes'
        return 'c:%d u:%d v:%d' % (self.c, self.u, self.v)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Define  value oriented equality, which is useful for testers'
        return isinstance(other, self.__class__) and self.c == other.c and (self.u == other.u) and (self.v == other.v)

    def __ne__(self, other):
        if False:
            return 10
        'Defined for consistency with __eq__'
        return not self == other

def multiset_partitions_taocp(multiplicities):
    if False:
        while True:
            i = 10
    "Enumerates partitions of a multiset.\n\n    Parameters\n    ==========\n\n    multiplicities\n         list of integer multiplicities of the components of the multiset.\n\n    Yields\n    ======\n\n    state\n        Internal data structure which encodes a particular partition.\n        This output is then usually processed by a visitor function\n        which combines the information from this data structure with\n        the components themselves to produce an actual partition.\n\n        Unless they wish to create their own visitor function, users will\n        have little need to look inside this data structure.  But, for\n        reference, it is a 3-element list with components:\n\n        f\n            is a frame array, which is used to divide pstack into parts.\n\n        lpart\n            points to the base of the topmost part.\n\n        pstack\n            is an array of PartComponent objects.\n\n        The ``state`` output offers a peek into the internal data\n        structures of the enumeration function.  The client should\n        treat this as read-only; any modification of the data\n        structure will cause unpredictable (and almost certainly\n        incorrect) results.  Also, the components of ``state`` are\n        modified in place at each iteration.  Hence, the visitor must\n        be called at each loop iteration.  Accumulating the ``state``\n        instances and processing them later will not work.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.enumerative import list_visitor\n    >>> from sympy.utilities.enumerative import multiset_partitions_taocp\n    >>> # variables components and multiplicities represent the multiset 'abb'\n    >>> components = 'ab'\n    >>> multiplicities = [1, 2]\n    >>> states = multiset_partitions_taocp(multiplicities)\n    >>> list(list_visitor(state, components) for state in states)\n    [[['a', 'b', 'b']],\n    [['a', 'b'], ['b']],\n    [['a'], ['b', 'b']],\n    [['a'], ['b'], ['b']]]\n\n    See Also\n    ========\n\n    sympy.utilities.iterables.multiset_partitions: Takes a multiset\n        as input and directly yields multiset partitions.  It\n        dispatches to a number of functions, including this one, for\n        implementation.  Most users will find it more convenient to\n        use than multiset_partitions_taocp.\n\n    "
    m = len(multiplicities)
    n = sum(multiplicities)
    pstack = [PartComponent() for i in range(n * m + 1)]
    f = [0] * (n + 1)
    for j in range(m):
        ps = pstack[j]
        ps.c = j
        ps.u = multiplicities[j]
        ps.v = multiplicities[j]
    f[0] = 0
    a = 0
    lpart = 0
    f[1] = m
    b = m
    while True:
        while True:
            j = a
            k = b
            x = False
            while j < b:
                pstack[k].u = pstack[j].u - pstack[j].v
                if pstack[k].u == 0:
                    x = True
                elif not x:
                    pstack[k].c = pstack[j].c
                    pstack[k].v = min(pstack[j].v, pstack[k].u)
                    x = pstack[k].u < pstack[j].v
                    k = k + 1
                else:
                    pstack[k].c = pstack[j].c
                    pstack[k].v = pstack[k].u
                    k = k + 1
                j = j + 1
            if k > b:
                a = b
                b = k
                lpart = lpart + 1
                f[lpart + 1] = b
            else:
                break
        state = [f, lpart, pstack]
        yield state
        while True:
            j = b - 1
            while pstack[j].v == 0:
                j = j - 1
            if j == a and pstack[j].v == 1:
                if lpart == 0:
                    return
                lpart = lpart - 1
                b = a
                a = f[lpart]
            else:
                pstack[j].v = pstack[j].v - 1
                for k in range(j + 1, b):
                    pstack[k].v = pstack[k].u
                break

def factoring_visitor(state, primes):
    if False:
        while True:
            i = 10
    'Use with multiset_partitions_taocp to enumerate the ways a\n    number can be expressed as a product of factors.  For this usage,\n    the exponents of the prime factors of a number are arguments to\n    the partition enumerator, while the corresponding prime factors\n    are input here.\n\n    Examples\n    ========\n\n    To enumerate the factorings of a number we can think of the elements of the\n    partition as being the prime factors and the multiplicities as being their\n    exponents.\n\n    >>> from sympy.utilities.enumerative import factoring_visitor\n    >>> from sympy.utilities.enumerative import multiset_partitions_taocp\n    >>> from sympy import factorint\n    >>> primes, multiplicities = zip(*factorint(24).items())\n    >>> primes\n    (2, 3)\n    >>> multiplicities\n    (3, 1)\n    >>> states = multiset_partitions_taocp(multiplicities)\n    >>> list(factoring_visitor(state, primes) for state in states)\n    [[24], [8, 3], [12, 2], [4, 6], [4, 2, 3], [6, 2, 2], [2, 2, 2, 3]]\n    '
    (f, lpart, pstack) = state
    factoring = []
    for i in range(lpart + 1):
        factor = 1
        for ps in pstack[f[i]:f[i + 1]]:
            if ps.v > 0:
                factor *= primes[ps.c] ** ps.v
        factoring.append(factor)
    return factoring

def list_visitor(state, components):
    if False:
        while True:
            i = 10
    "Return a list of lists to represent the partition.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.enumerative import list_visitor\n    >>> from sympy.utilities.enumerative import multiset_partitions_taocp\n    >>> states = multiset_partitions_taocp([1, 2, 1])\n    >>> s = next(states)\n    >>> list_visitor(s, 'abc')  # for multiset 'a b b c'\n    [['a', 'b', 'b', 'c']]\n    >>> s = next(states)\n    >>> list_visitor(s, [1, 2, 3])  # for multiset '1 2 2 3\n    [[1, 2, 2], [3]]\n    "
    (f, lpart, pstack) = state
    partition = []
    for i in range(lpart + 1):
        part = []
        for ps in pstack[f[i]:f[i + 1]]:
            if ps.v > 0:
                part.extend([components[ps.c]] * ps.v)
        partition.append(part)
    return partition

class MultisetPartitionTraverser:
    """
    Has methods to ``enumerate`` and ``count`` the partitions of a multiset.

    This implements a refactored and extended version of Knuth's algorithm
    7.1.2.5M [AOCP]_."

    The enumeration methods of this class are generators and return
    data structures which can be interpreted by the same visitor
    functions used for the output of ``multiset_partitions_taocp``.

    Examples
    ========

    >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
    >>> m = MultisetPartitionTraverser()
    >>> m.count_partitions([4,4,4,2])
    127750
    >>> m.count_partitions([3,3,3])
    686

    See Also
    ========

    multiset_partitions_taocp
    sympy.utilities.iterables.multiset_partitions

    References
    ==========

    .. [AOCP] Algorithm 7.1.2.5M in Volume 4A, Combinatoral Algorithms,
           Part 1, of The Art of Computer Programming, by Donald Knuth.

    .. [Factorisatio] On a Problem of Oppenheim concerning
           "Factorisatio Numerorum" E. R. Canfield, Paul Erdos, Carl
           Pomerance, JOURNAL OF NUMBER THEORY, Vol. 17, No. 1. August
           1983.  See section 7 for a description of an algorithm
           similar to Knuth's.

    .. [Yorgey] Generating Multiset Partitions, Brent Yorgey, The
           Monad.Reader, Issue 8, September 2007.

    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.debug = False
        self.k1 = 0
        self.k2 = 0
        self.p1 = 0
        self.pstack = None
        self.f = None
        self.lpart = 0
        self.discarded = 0
        self.dp_stack = []
        if not hasattr(self, 'dp_map'):
            self.dp_map = {}

    def db_trace(self, msg):
        if False:
            while True:
                i = 10
        'Useful for understanding/debugging the algorithms.  Not\n        generally activated in end-user code.'
        if self.debug:
            raise RuntimeError

    def _initialize_enumeration(self, multiplicities):
        if False:
            for i in range(10):
                print('nop')
        'Allocates and initializes the partition stack.\n\n        This is called from the enumeration/counting routines, so\n        there is no need to call it separately.'
        num_components = len(multiplicities)
        cardinality = sum(multiplicities)
        self.pstack = [PartComponent() for i in range(num_components * cardinality + 1)]
        self.f = [0] * (cardinality + 1)
        for j in range(num_components):
            ps = self.pstack[j]
            ps.c = j
            ps.u = multiplicities[j]
            ps.v = multiplicities[j]
        self.f[0] = 0
        self.f[1] = num_components
        self.lpart = 0

    def decrement_part(self, part):
        if False:
            print('Hello World!')
        'Decrements part (a subrange of pstack), if possible, returning\n        True iff the part was successfully decremented.\n\n        If you think of the v values in the part as a multi-digit\n        integer (least significant digit on the right) this is\n        basically decrementing that integer, but with the extra\n        constraint that the leftmost digit cannot be decremented to 0.\n\n        Parameters\n        ==========\n\n        part\n           The part, represented as a list of PartComponent objects,\n           which is to be decremented.\n\n        '
        plen = len(part)
        for j in range(plen - 1, -1, -1):
            if j == 0 and part[j].v > 1 or (j > 0 and part[j].v > 0):
                part[j].v -= 1
                for k in range(j + 1, plen):
                    part[k].v = part[k].u
                return True
        return False

    def decrement_part_small(self, part, ub):
        if False:
            i = 10
            return i + 15
        "Decrements part (a subrange of pstack), if possible, returning\n        True iff the part was successfully decremented.\n\n        Parameters\n        ==========\n\n        part\n            part to be decremented (topmost part on the stack)\n\n        ub\n            the maximum number of parts allowed in a partition\n            returned by the calling traversal.\n\n        Notes\n        =====\n\n        The goal of this modification of the ordinary decrement method\n        is to fail (meaning that the subtree rooted at this part is to\n        be skipped) when it can be proved that this part can only have\n        child partitions which are larger than allowed by ``ub``. If a\n        decision is made to fail, it must be accurate, otherwise the\n        enumeration will miss some partitions.  But, it is OK not to\n        capture all the possible failures -- if a part is passed that\n        should not be, the resulting too-large partitions are filtered\n        by the enumeration one level up.  However, as is usual in\n        constrained enumerations, failing early is advantageous.\n\n        The tests used by this method catch the most common cases,\n        although this implementation is by no means the last word on\n        this problem.  The tests include:\n\n        1) ``lpart`` must be less than ``ub`` by at least 2.  This is because\n           once a part has been decremented, the partition\n           will gain at least one child in the spread step.\n\n        2) If the leading component of the part is about to be\n           decremented, check for how many parts will be added in\n           order to use up the unallocated multiplicity in that\n           leading component, and fail if this number is greater than\n           allowed by ``ub``.  (See code for the exact expression.)  This\n           test is given in the answer to Knuth's problem 7.2.1.5.69.\n\n        3) If there is *exactly* enough room to expand the leading\n           component by the above test, check the next component (if\n           it exists) once decrementing has finished.  If this has\n           ``v == 0``, this next component will push the expansion over the\n           limit by 1, so fail.\n        "
        if self.lpart >= ub - 1:
            self.p1 += 1
            return False
        plen = len(part)
        for j in range(plen - 1, -1, -1):
            if j == 0 and (part[0].v - 1) * (ub - self.lpart) < part[0].u:
                self.k1 += 1
                return False
            if j == 0 and part[j].v > 1 or (j > 0 and part[j].v > 0):
                part[j].v -= 1
                for k in range(j + 1, plen):
                    part[k].v = part[k].u
                if plen > 1 and part[1].v == 0 and (part[0].u - part[0].v == (ub - self.lpart - 1) * part[0].v):
                    self.k2 += 1
                    self.db_trace('Decrement fails test 3')
                    return False
                return True
        return False

    def decrement_part_large(self, part, amt, lb):
        if False:
            return 10
        'Decrements part, while respecting size constraint.\n\n        A part can have no children which are of sufficient size (as\n        indicated by ``lb``) unless that part has sufficient\n        unallocated multiplicity.  When enforcing the size constraint,\n        this method will decrement the part (if necessary) by an\n        amount needed to ensure sufficient unallocated multiplicity.\n\n        Returns True iff the part was successfully decremented.\n\n        Parameters\n        ==========\n\n        part\n            part to be decremented (topmost part on the stack)\n\n        amt\n            Can only take values 0 or 1.  A value of 1 means that the\n            part must be decremented, and then the size constraint is\n            enforced.  A value of 0 means just to enforce the ``lb``\n            size constraint.\n\n        lb\n            The partitions produced by the calling enumeration must\n            have more parts than this value.\n\n        '
        if amt == 1:
            if not self.decrement_part(part):
                return False
        min_unalloc = lb - self.lpart
        if min_unalloc <= 0:
            return True
        total_mult = sum((pc.u for pc in part))
        total_alloc = sum((pc.v for pc in part))
        if total_mult <= min_unalloc:
            return False
        deficit = min_unalloc - (total_mult - total_alloc)
        if deficit <= 0:
            return True
        for i in range(len(part) - 1, -1, -1):
            if i == 0:
                if part[0].v > deficit:
                    part[0].v -= deficit
                    return True
                else:
                    return False
            elif part[i].v >= deficit:
                part[i].v -= deficit
                return True
            else:
                deficit -= part[i].v
                part[i].v = 0

    def decrement_part_range(self, part, lb, ub):
        if False:
            return 10
        'Decrements part (a subrange of pstack), if possible, returning\n        True iff the part was successfully decremented.\n\n        Parameters\n        ==========\n\n        part\n            part to be decremented (topmost part on the stack)\n\n        ub\n            the maximum number of parts allowed in a partition\n            returned by the calling traversal.\n\n        lb\n            The partitions produced by the calling enumeration must\n            have more parts than this value.\n\n        Notes\n        =====\n\n        Combines the constraints of _small and _large decrement\n        methods.  If returns success, part has been decremented at\n        least once, but perhaps by quite a bit more if needed to meet\n        the lb constraint.\n        '
        return self.decrement_part_small(part, ub) and self.decrement_part_large(part, 0, lb)

    def spread_part_multiplicity(self):
        if False:
            return 10
        'Returns True if a new part has been created, and\n        adjusts pstack, f and lpart as needed.\n\n        Notes\n        =====\n\n        Spreads unallocated multiplicity from the current top part\n        into a new part created above the current on the stack.  This\n        new part is constrained to be less than or equal to the old in\n        terms of the part ordering.\n\n        This call does nothing (and returns False) if the current top\n        part has no unallocated multiplicity.\n\n        '
        j = self.f[self.lpart]
        k = self.f[self.lpart + 1]
        base = k
        changed = False
        for j in range(self.f[self.lpart], self.f[self.lpart + 1]):
            self.pstack[k].u = self.pstack[j].u - self.pstack[j].v
            if self.pstack[k].u == 0:
                changed = True
            else:
                self.pstack[k].c = self.pstack[j].c
                if changed:
                    self.pstack[k].v = self.pstack[k].u
                elif self.pstack[k].u < self.pstack[j].v:
                    self.pstack[k].v = self.pstack[k].u
                    changed = True
                else:
                    self.pstack[k].v = self.pstack[j].v
                k = k + 1
        if k > base:
            self.lpart = self.lpart + 1
            self.f[self.lpart + 1] = k
            return True
        return False

    def top_part(self):
        if False:
            return 10
        'Return current top part on the stack, as a slice of pstack.\n\n        '
        return self.pstack[self.f[self.lpart]:self.f[self.lpart + 1]]

    def enum_all(self, multiplicities):
        if False:
            for i in range(10):
                print('nop')
        "Enumerate the partitions of a multiset.\n\n        Examples\n        ========\n\n        >>> from sympy.utilities.enumerative import list_visitor\n        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser\n        >>> m = MultisetPartitionTraverser()\n        >>> states = m.enum_all([2,2])\n        >>> list(list_visitor(state, 'ab') for state in states)\n        [[['a', 'a', 'b', 'b']],\n        [['a', 'a', 'b'], ['b']],\n        [['a', 'a'], ['b', 'b']],\n        [['a', 'a'], ['b'], ['b']],\n        [['a', 'b', 'b'], ['a']],\n        [['a', 'b'], ['a', 'b']],\n        [['a', 'b'], ['a'], ['b']],\n        [['a'], ['a'], ['b', 'b']],\n        [['a'], ['a'], ['b'], ['b']]]\n\n        See Also\n        ========\n\n        multiset_partitions_taocp:\n            which provides the same result as this method, but is\n            about twice as fast.  Hence, enum_all is primarily useful\n            for testing.  Also see the function for a discussion of\n            states and visitors.\n\n        "
        self._initialize_enumeration(multiplicities)
        while True:
            while self.spread_part_multiplicity():
                pass
            state = [self.f, self.lpart, self.pstack]
            yield state
            while not self.decrement_part(self.top_part()):
                if self.lpart == 0:
                    return
                self.lpart -= 1

    def enum_small(self, multiplicities, ub):
        if False:
            print('Hello World!')
        "Enumerate multiset partitions with no more than ``ub`` parts.\n\n        Equivalent to enum_range(multiplicities, 0, ub)\n\n        Parameters\n        ==========\n\n        multiplicities\n             list of multiplicities of the components of the multiset.\n\n        ub\n            Maximum number of parts\n\n        Examples\n        ========\n\n        >>> from sympy.utilities.enumerative import list_visitor\n        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser\n        >>> m = MultisetPartitionTraverser()\n        >>> states = m.enum_small([2,2], 2)\n        >>> list(list_visitor(state, 'ab') for state in states)\n        [[['a', 'a', 'b', 'b']],\n        [['a', 'a', 'b'], ['b']],\n        [['a', 'a'], ['b', 'b']],\n        [['a', 'b', 'b'], ['a']],\n        [['a', 'b'], ['a', 'b']]]\n\n        The implementation is based, in part, on the answer given to\n        exercise 69, in Knuth [AOCP]_.\n\n        See Also\n        ========\n\n        enum_all, enum_large, enum_range\n\n        "
        self.discarded = 0
        if ub <= 0:
            return
        self._initialize_enumeration(multiplicities)
        while True:
            while self.spread_part_multiplicity():
                self.db_trace('spread 1')
                if self.lpart >= ub:
                    self.discarded += 1
                    self.db_trace('  Discarding')
                    self.lpart = ub - 2
                    break
            else:
                state = [self.f, self.lpart, self.pstack]
                yield state
            while not self.decrement_part_small(self.top_part(), ub):
                self.db_trace('Failed decrement, going to backtrack')
                if self.lpart == 0:
                    return
                self.lpart -= 1
                self.db_trace('Backtracked to')
            self.db_trace('decrement ok, about to expand')

    def enum_large(self, multiplicities, lb):
        if False:
            i = 10
            return i + 15
        "Enumerate the partitions of a multiset with lb < num(parts)\n\n        Equivalent to enum_range(multiplicities, lb, sum(multiplicities))\n\n        Parameters\n        ==========\n\n        multiplicities\n            list of multiplicities of the components of the multiset.\n\n        lb\n            Number of parts in the partition must be greater than\n            this lower bound.\n\n\n        Examples\n        ========\n\n        >>> from sympy.utilities.enumerative import list_visitor\n        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser\n        >>> m = MultisetPartitionTraverser()\n        >>> states = m.enum_large([2,2], 2)\n        >>> list(list_visitor(state, 'ab') for state in states)\n        [[['a', 'a'], ['b'], ['b']],\n        [['a', 'b'], ['a'], ['b']],\n        [['a'], ['a'], ['b', 'b']],\n        [['a'], ['a'], ['b'], ['b']]]\n\n        See Also\n        ========\n\n        enum_all, enum_small, enum_range\n\n        "
        self.discarded = 0
        if lb >= sum(multiplicities):
            return
        self._initialize_enumeration(multiplicities)
        self.decrement_part_large(self.top_part(), 0, lb)
        while True:
            good_partition = True
            while self.spread_part_multiplicity():
                if not self.decrement_part_large(self.top_part(), 0, lb):
                    self.discarded += 1
                    good_partition = False
                    break
            if good_partition:
                state = [self.f, self.lpart, self.pstack]
                yield state
            while not self.decrement_part_large(self.top_part(), 1, lb):
                if self.lpart == 0:
                    return
                self.lpart -= 1

    def enum_range(self, multiplicities, lb, ub):
        if False:
            for i in range(10):
                print('nop')
        "Enumerate the partitions of a multiset with\n        ``lb < num(parts) <= ub``.\n\n        In particular, if partitions with exactly ``k`` parts are\n        desired, call with ``(multiplicities, k - 1, k)``.  This\n        method generalizes enum_all, enum_small, and enum_large.\n\n        Examples\n        ========\n\n        >>> from sympy.utilities.enumerative import list_visitor\n        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser\n        >>> m = MultisetPartitionTraverser()\n        >>> states = m.enum_range([2,2], 1, 2)\n        >>> list(list_visitor(state, 'ab') for state in states)\n        [[['a', 'a', 'b'], ['b']],\n        [['a', 'a'], ['b', 'b']],\n        [['a', 'b', 'b'], ['a']],\n        [['a', 'b'], ['a', 'b']]]\n\n        "
        self.discarded = 0
        if ub <= 0 or lb >= sum(multiplicities):
            return
        self._initialize_enumeration(multiplicities)
        self.decrement_part_large(self.top_part(), 0, lb)
        while True:
            good_partition = True
            while self.spread_part_multiplicity():
                self.db_trace('spread 1')
                if not self.decrement_part_large(self.top_part(), 0, lb):
                    self.db_trace('  Discarding (large cons)')
                    self.discarded += 1
                    good_partition = False
                    break
                elif self.lpart >= ub:
                    self.discarded += 1
                    good_partition = False
                    self.db_trace('  Discarding small cons')
                    self.lpart = ub - 2
                    break
            if good_partition:
                state = [self.f, self.lpart, self.pstack]
                yield state
            while not self.decrement_part_range(self.top_part(), lb, ub):
                self.db_trace('Failed decrement, going to backtrack')
                if self.lpart == 0:
                    return
                self.lpart -= 1
                self.db_trace('Backtracked to')
            self.db_trace('decrement ok, about to expand')

    def count_partitions_slow(self, multiplicities):
        if False:
            return 10
        'Returns the number of partitions of a multiset whose elements\n        have the multiplicities given in ``multiplicities``.\n\n        Primarily for comparison purposes.  It follows the same path as\n        enumerate, and counts, rather than generates, the partitions.\n\n        See Also\n        ========\n\n        count_partitions\n            Has the same calling interface, but is much faster.\n\n        '
        self.pcount = 0
        self._initialize_enumeration(multiplicities)
        while True:
            while self.spread_part_multiplicity():
                pass
            self.pcount += 1
            while not self.decrement_part(self.top_part()):
                if self.lpart == 0:
                    return self.pcount
                self.lpart -= 1

    def count_partitions(self, multiplicities):
        if False:
            for i in range(10):
                print('nop')
        "Returns the number of partitions of a multiset whose components\n        have the multiplicities given in ``multiplicities``.\n\n        For larger counts, this method is much faster than calling one\n        of the enumerators and counting the result.  Uses dynamic\n        programming to cut down on the number of nodes actually\n        explored.  The dictionary used in order to accelerate the\n        counting process is stored in the ``MultisetPartitionTraverser``\n        object and persists across calls.  If the user does not\n        expect to call ``count_partitions`` for any additional\n        multisets, the object should be cleared to save memory.  On\n        the other hand, the cache built up from one count run can\n        significantly speed up subsequent calls to ``count_partitions``,\n        so it may be advantageous not to clear the object.\n\n        Examples\n        ========\n\n        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser\n        >>> m = MultisetPartitionTraverser()\n        >>> m.count_partitions([9,8,2])\n        288716\n        >>> m.count_partitions([2,2])\n        9\n        >>> del m\n\n        Notes\n        =====\n\n        If one looks at the workings of Knuth's algorithm M [AOCP]_, it\n        can be viewed as a traversal of a binary tree of parts.  A\n        part has (up to) two children, the left child resulting from\n        the spread operation, and the right child from the decrement\n        operation.  The ordinary enumeration of multiset partitions is\n        an in-order traversal of this tree, and with the partitions\n        corresponding to paths from the root to the leaves. The\n        mapping from paths to partitions is a little complicated,\n        since the partition would contain only those parts which are\n        leaves or the parents of a spread link, not those which are\n        parents of a decrement link.\n\n        For counting purposes, it is sufficient to count leaves, and\n        this can be done with a recursive in-order traversal.  The\n        number of leaves of a subtree rooted at a particular part is a\n        function only of that part itself, so memoizing has the\n        potential to speed up the counting dramatically.\n\n        This method follows a computational approach which is similar\n        to the hypothetical memoized recursive function, but with two\n        differences:\n\n        1) This method is iterative, borrowing its structure from the\n           other enumerations and maintaining an explicit stack of\n           parts which are in the process of being counted.  (There\n           may be multisets which can be counted reasonably quickly by\n           this implementation, but which would overflow the default\n           Python recursion limit with a recursive implementation.)\n\n        2) Instead of using the part data structure directly, a more\n           compact key is constructed.  This saves space, but more\n           importantly coalesces some parts which would remain\n           separate with physical keys.\n\n        Unlike the enumeration functions, there is currently no _range\n        version of count_partitions.  If someone wants to stretch\n        their brain, it should be possible to construct one by\n        memoizing with a histogram of counts rather than a single\n        count, and combining the histograms.\n        "
        self.pcount = 0
        self.dp_stack = []
        self._initialize_enumeration(multiplicities)
        pkey = part_key(self.top_part())
        self.dp_stack.append([(pkey, 0)])
        while True:
            while self.spread_part_multiplicity():
                pkey = part_key(self.top_part())
                if pkey in self.dp_map:
                    self.pcount += self.dp_map[pkey] - 1
                    self.lpart -= 1
                    break
                else:
                    self.dp_stack.append([(pkey, self.pcount)])
            self.pcount += 1
            while not self.decrement_part(self.top_part()):
                for (key, oldcount) in self.dp_stack.pop():
                    self.dp_map[key] = self.pcount - oldcount
                if self.lpart == 0:
                    return self.pcount
                self.lpart -= 1
            pkey = part_key(self.top_part())
            self.dp_stack[-1].append((pkey, self.pcount))

def part_key(part):
    if False:
        while True:
            i = 10
    'Helper for MultisetPartitionTraverser.count_partitions that\n    creates a key for ``part``, that only includes information which can\n    affect the count for that part.  (Any irrelevant information just\n    reduces the effectiveness of dynamic programming.)\n\n    Notes\n    =====\n\n    This member function is a candidate for future exploration. There\n    are likely symmetries that can be exploited to coalesce some\n    ``part_key`` values, and thereby save space and improve\n    performance.\n\n    '
    rval = []
    for ps in part:
        rval.append(ps.u)
        rval.append(ps.v)
    return tuple(rval)