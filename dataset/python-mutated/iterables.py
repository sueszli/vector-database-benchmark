from collections import Counter, defaultdict, OrderedDict
from itertools import chain, combinations, combinations_with_replacement, cycle, islice, permutations, product, groupby
from itertools import product as cartes
from operator import gt
from sympy.utilities.enumerative import multiset_partitions_taocp, list_visitor, MultisetPartitionTraverser
from sympy.utilities.misc import as_int
from sympy.utilities.decorator import deprecated

def is_palindromic(s, i=0, j=None):
    if False:
        return 10
    "\n    Return True if the sequence is the same from left to right as it\n    is from right to left in the whole sequence (default) or in the\n    Python slice ``s[i: j]``; else False.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import is_palindromic\n    >>> is_palindromic([1, 0, 1])\n    True\n    >>> is_palindromic('abcbb')\n    False\n    >>> is_palindromic('abcbb', 1)\n    False\n\n    Normal Python slicing is performed in place so there is no need to\n    create a slice of the sequence for testing:\n\n    >>> is_palindromic('abcbb', 1, -1)\n    True\n    >>> is_palindromic('abcbb', -4, -1)\n    True\n\n    See Also\n    ========\n\n    sympy.ntheory.digits.is_palindromic: tests integers\n\n    "
    (i, j, _) = slice(i, j).indices(len(s))
    m = (j - i) // 2
    return all((s[i + k] == s[j - 1 - k] for k in range(m)))

def flatten(iterable, levels=None, cls=None):
    if False:
        while True:
            i = 10
    '\n    Recursively denest iterable containers.\n\n    >>> from sympy import flatten\n\n    >>> flatten([1, 2, 3])\n    [1, 2, 3]\n    >>> flatten([1, 2, [3]])\n    [1, 2, 3]\n    >>> flatten([1, [2, 3], [4, 5]])\n    [1, 2, 3, 4, 5]\n    >>> flatten([1.0, 2, (1, None)])\n    [1.0, 2, 1, None]\n\n    If you want to denest only a specified number of levels of\n    nested containers, then set ``levels`` flag to the desired\n    number of levels::\n\n    >>> ls = [[(-2, -1), (1, 2)], [(0, 0)]]\n\n    >>> flatten(ls, levels=1)\n    [(-2, -1), (1, 2), (0, 0)]\n\n    If cls argument is specified, it will only flatten instances of that\n    class, for example:\n\n    >>> from sympy import Basic, S\n    >>> class MyOp(Basic):\n    ...     pass\n    ...\n    >>> flatten([MyOp(S(1), MyOp(S(2), S(3)))], cls=MyOp)\n    [1, 2, 3]\n\n    adapted from https://kogs-www.informatik.uni-hamburg.de/~meine/python_tricks\n    '
    from sympy.tensor.array import NDimArray
    if levels is not None:
        if not levels:
            return iterable
        elif levels > 0:
            levels -= 1
        else:
            raise ValueError('expected non-negative number of levels, got %s' % levels)
    if cls is None:

        def reducible(x):
            if False:
                i = 10
                return i + 15
            return is_sequence(x, set)
    else:

        def reducible(x):
            if False:
                for i in range(10):
                    print('nop')
            return isinstance(x, cls)
    result = []
    for el in iterable:
        if reducible(el):
            if hasattr(el, 'args') and (not isinstance(el, NDimArray)):
                el = el.args
            result.extend(flatten(el, levels=levels, cls=cls))
        else:
            result.append(el)
    return result

def unflatten(iter, n=2):
    if False:
        print('Hello World!')
    'Group ``iter`` into tuples of length ``n``. Raise an error if\n    the length of ``iter`` is not a multiple of ``n``.\n    '
    if n < 1 or len(iter) % n:
        raise ValueError('iter length is not a multiple of %i' % n)
    return list(zip(*(iter[i::n] for i in range(n))))

def reshape(seq, how):
    if False:
        print('Hello World!')
    'Reshape the sequence according to the template in ``how``.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities import reshape\n    >>> seq = list(range(1, 9))\n\n    >>> reshape(seq, [4]) # lists of 4\n    [[1, 2, 3, 4], [5, 6, 7, 8]]\n\n    >>> reshape(seq, (4,)) # tuples of 4\n    [(1, 2, 3, 4), (5, 6, 7, 8)]\n\n    >>> reshape(seq, (2, 2)) # tuples of 4\n    [(1, 2, 3, 4), (5, 6, 7, 8)]\n\n    >>> reshape(seq, (2, [2])) # (i, i, [i, i])\n    [(1, 2, [3, 4]), (5, 6, [7, 8])]\n\n    >>> reshape(seq, ((2,), [2])) # etc....\n    [((1, 2), [3, 4]), ((5, 6), [7, 8])]\n\n    >>> reshape(seq, (1, [2], 1))\n    [(1, [2, 3], 4), (5, [6, 7], 8)]\n\n    >>> reshape(tuple(seq), ([[1], 1, (2,)],))\n    (([[1], 2, (3, 4)],), ([[5], 6, (7, 8)],))\n\n    >>> reshape(tuple(seq), ([1], 1, (2,)))\n    (([1], 2, (3, 4)), ([5], 6, (7, 8)))\n\n    >>> reshape(list(range(12)), [2, [3], {2}, (1, (3,), 1)])\n    [[0, 1, [2, 3, 4], {5, 6}, (7, (8, 9, 10), 11)]]\n\n    '
    m = sum(flatten(how))
    (n, rem) = divmod(len(seq), m)
    if m < 0 or rem:
        raise ValueError('template must sum to positive number that divides the length of the sequence')
    i = 0
    container = type(how)
    rv = [None] * n
    for k in range(len(rv)):
        _rv = []
        for hi in how:
            if isinstance(hi, int):
                _rv.extend(seq[i:i + hi])
                i += hi
            else:
                n = sum(flatten(hi))
                hi_type = type(hi)
                _rv.append(hi_type(reshape(seq[i:i + n], hi)[0]))
                i += n
        rv[k] = container(_rv)
    return type(seq)(rv)

def group(seq, multiple=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Splits a sequence into a list of lists of equal, adjacent elements.\n\n    Examples\n    ========\n\n    >>> from sympy import group\n\n    >>> group([1, 1, 1, 2, 2, 3])\n    [[1, 1, 1], [2, 2], [3]]\n    >>> group([1, 1, 1, 2, 2, 3], multiple=False)\n    [(1, 3), (2, 2), (3, 1)]\n    >>> group([1, 1, 3, 2, 2, 1], multiple=False)\n    [(1, 2), (3, 1), (2, 2), (1, 1)]\n\n    See Also\n    ========\n\n    multiset\n\n    '
    if multiple:
        return [list(g) for (_, g) in groupby(seq)]
    return [(k, len(list(g))) for (k, g) in groupby(seq)]

def _iproduct2(iterable1, iterable2):
    if False:
        for i in range(10):
            print('nop')
    'Cartesian product of two possibly infinite iterables'
    it1 = iter(iterable1)
    it2 = iter(iterable2)
    elems1 = []
    elems2 = []
    sentinel = object()

    def append(it, elems):
        if False:
            for i in range(10):
                print('nop')
        e = next(it, sentinel)
        if e is not sentinel:
            elems.append(e)
    n = 0
    append(it1, elems1)
    append(it2, elems2)
    while n <= len(elems1) + len(elems2):
        for m in range(n - len(elems1) + 1, len(elems2)):
            yield (elems1[n - m], elems2[m])
        n += 1
        append(it1, elems1)
        append(it2, elems2)

def iproduct(*iterables):
    if False:
        print('Hello World!')
    '\n    Cartesian product of iterables.\n\n    Generator of the Cartesian product of iterables. This is analogous to\n    itertools.product except that it works with infinite iterables and will\n    yield any item from the infinite product eventually.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import iproduct\n    >>> sorted(iproduct([1,2], [3,4]))\n    [(1, 3), (1, 4), (2, 3), (2, 4)]\n\n    With an infinite iterator:\n\n    >>> from sympy import S\n    >>> (3,) in iproduct(S.Integers)\n    True\n    >>> (3, 4) in iproduct(S.Integers, S.Integers)\n    True\n\n    .. seealso::\n\n       `itertools.product\n       <https://docs.python.org/3/library/itertools.html#itertools.product>`_\n    '
    if len(iterables) == 0:
        yield ()
        return
    elif len(iterables) == 1:
        for e in iterables[0]:
            yield (e,)
    elif len(iterables) == 2:
        yield from _iproduct2(*iterables)
    else:
        (first, others) = (iterables[0], iterables[1:])
        for (ef, eo) in _iproduct2(first, iproduct(*others)):
            yield ((ef,) + eo)

def multiset(seq):
    if False:
        return 10
    "Return the hashable sequence in multiset form with values being the\n    multiplicity of the item in the sequence.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import multiset\n    >>> multiset('mississippi')\n    {'i': 4, 'm': 1, 'p': 2, 's': 4}\n\n    See Also\n    ========\n\n    group\n\n    "
    return dict(Counter(seq).items())

def ibin(n, bits=None, str=False):
    if False:
        print('Hello World!')
    "Return a list of length ``bits`` corresponding to the binary value\n    of ``n`` with small bits to the right (last). If bits is omitted, the\n    length will be the number required to represent ``n``. If the bits are\n    desired in reversed order, use the ``[::-1]`` slice of the returned list.\n\n    If a sequence of all bits-length lists starting from ``[0, 0,..., 0]``\n    through ``[1, 1, ..., 1]`` are desired, pass a non-integer for bits, e.g.\n    ``'all'``.\n\n    If the bit *string* is desired pass ``str=True``.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import ibin\n    >>> ibin(2)\n    [1, 0]\n    >>> ibin(2, 4)\n    [0, 0, 1, 0]\n\n    If all lists corresponding to 0 to 2**n - 1, pass a non-integer\n    for bits:\n\n    >>> bits = 2\n    >>> for i in ibin(2, 'all'):\n    ...     print(i)\n    (0, 0)\n    (0, 1)\n    (1, 0)\n    (1, 1)\n\n    If a bit string is desired of a given length, use str=True:\n\n    >>> n = 123\n    >>> bits = 10\n    >>> ibin(n, bits, str=True)\n    '0001111011'\n    >>> ibin(n, bits, str=True)[::-1]  # small bits left\n    '1101111000'\n    >>> list(ibin(3, 'all', str=True))\n    ['000', '001', '010', '011', '100', '101', '110', '111']\n\n    "
    if n < 0:
        raise ValueError('negative numbers are not allowed')
    n = as_int(n)
    if bits is None:
        bits = 0
    else:
        try:
            bits = as_int(bits)
        except ValueError:
            bits = -1
        else:
            if n.bit_length() > bits:
                raise ValueError('`bits` must be >= {}'.format(n.bit_length()))
    if not str:
        if bits >= 0:
            return [1 if i == '1' else 0 for i in bin(n)[2:].rjust(bits, '0')]
        else:
            return variations(range(2), n, repetition=True)
    elif bits >= 0:
        return bin(n)[2:].rjust(bits, '0')
    else:
        return (bin(i)[2:].rjust(n, '0') for i in range(2 ** n))

def variations(seq, n, repetition=False):
    if False:
        return 10
    "Returns an iterator over the n-sized variations of ``seq`` (size N).\n    ``repetition`` controls whether items in ``seq`` can appear more than once;\n\n    Examples\n    ========\n\n    ``variations(seq, n)`` will return `\\frac{N!}{(N - n)!}` permutations without\n    repetition of ``seq``'s elements:\n\n        >>> from sympy import variations\n        >>> list(variations([1, 2], 2))\n        [(1, 2), (2, 1)]\n\n    ``variations(seq, n, True)`` will return the `N^n` permutations obtained\n    by allowing repetition of elements:\n\n        >>> list(variations([1, 2], 2, repetition=True))\n        [(1, 1), (1, 2), (2, 1), (2, 2)]\n\n    If you ask for more items than are in the set you get the empty set unless\n    you allow repetitions:\n\n        >>> list(variations([0, 1], 3, repetition=False))\n        []\n        >>> list(variations([0, 1], 3, repetition=True))[:4]\n        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]\n\n    .. seealso::\n\n       `itertools.permutations\n       <https://docs.python.org/3/library/itertools.html#itertools.permutations>`_,\n       `itertools.product\n       <https://docs.python.org/3/library/itertools.html#itertools.product>`_\n    "
    if not repetition:
        seq = tuple(seq)
        if len(seq) < n:
            return iter(())
        return permutations(seq, n)
    elif n == 0:
        return iter(((),))
    else:
        return product(seq, repeat=n)

def subsets(seq, k=None, repetition=False):
    if False:
        print('Hello World!')
    'Generates all `k`-subsets (combinations) from an `n`-element set, ``seq``.\n\n    A `k`-subset of an `n`-element set is any subset of length exactly `k`. The\n    number of `k`-subsets of an `n`-element set is given by ``binomial(n, k)``,\n    whereas there are `2^n` subsets all together. If `k` is ``None`` then all\n    `2^n` subsets will be returned from shortest to longest.\n\n    Examples\n    ========\n\n    >>> from sympy import subsets\n\n    ``subsets(seq, k)`` will return the\n    `\\frac{n!}{k!(n - k)!}` `k`-subsets (combinations)\n    without repetition, i.e. once an item has been removed, it can no\n    longer be "taken":\n\n        >>> list(subsets([1, 2], 2))\n        [(1, 2)]\n        >>> list(subsets([1, 2]))\n        [(), (1,), (2,), (1, 2)]\n        >>> list(subsets([1, 2, 3], 2))\n        [(1, 2), (1, 3), (2, 3)]\n\n\n    ``subsets(seq, k, repetition=True)`` will return the\n    `\\frac{(n - 1 + k)!}{k!(n - 1)!}`\n    combinations *with* repetition:\n\n        >>> list(subsets([1, 2], 2, repetition=True))\n        [(1, 1), (1, 2), (2, 2)]\n\n    If you ask for more items than are in the set you get the empty set unless\n    you allow repetitions:\n\n        >>> list(subsets([0, 1], 3, repetition=False))\n        []\n        >>> list(subsets([0, 1], 3, repetition=True))\n        [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]\n\n    '
    if k is None:
        if not repetition:
            return chain.from_iterable((combinations(seq, k) for k in range(len(seq) + 1)))
        else:
            return chain.from_iterable((combinations_with_replacement(seq, k) for k in range(len(seq) + 1)))
    elif not repetition:
        return combinations(seq, k)
    else:
        return combinations_with_replacement(seq, k)

def filter_symbols(iterator, exclude):
    if False:
        i = 10
        return i + 15
    '\n    Only yield elements from `iterator` that do not occur in `exclude`.\n\n    Parameters\n    ==========\n\n    iterator : iterable\n        iterator to take elements from\n\n    exclude : iterable\n        elements to exclude\n\n    Returns\n    =======\n\n    iterator : iterator\n        filtered iterator\n    '
    exclude = set(exclude)
    for s in iterator:
        if s not in exclude:
            yield s

def numbered_symbols(prefix='x', cls=None, start=0, exclude=(), *args, **assumptions):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generate an infinite stream of Symbols consisting of a prefix and\n    increasing subscripts provided that they do not occur in ``exclude``.\n\n    Parameters\n    ==========\n\n    prefix : str, optional\n        The prefix to use. By default, this function will generate symbols of\n        the form "x0", "x1", etc.\n\n    cls : class, optional\n        The class to use. By default, it uses ``Symbol``, but you can also use ``Wild``\n        or ``Dummy``.\n\n    start : int, optional\n        The start number.  By default, it is 0.\n\n    exclude : list, tuple, set of cls, optional\n        Symbols to be excluded.\n\n    *args, **kwargs\n        Additional positional and keyword arguments are passed to the *cls* class.\n\n    Returns\n    =======\n\n    sym : Symbol\n        The subscripted symbols.\n    '
    exclude = set(exclude or [])
    if cls is None:
        from sympy.core import Symbol
        cls = Symbol
    while True:
        name = '%s%s' % (prefix, start)
        s = cls(name, *args, **assumptions)
        if s not in exclude:
            yield s
        start += 1

def capture(func):
    if False:
        i = 10
        return i + 15
    "Return the printed output of func().\n\n    ``func`` should be a function without arguments that produces output with\n    print statements.\n\n    >>> from sympy.utilities.iterables import capture\n    >>> from sympy import pprint\n    >>> from sympy.abc import x\n    >>> def foo():\n    ...     print('hello world!')\n    ...\n    >>> 'hello' in capture(foo) # foo, not foo()\n    True\n    >>> capture(lambda: pprint(2/x))\n    '2\\n-\\nx\\n'\n\n    "
    from io import StringIO
    import sys
    stdout = sys.stdout
    sys.stdout = file = StringIO()
    try:
        func()
    finally:
        sys.stdout = stdout
    return file.getvalue()

def sift(seq, keyfunc, binary=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Sift the sequence, ``seq`` according to ``keyfunc``.\n\n    Returns\n    =======\n\n    When ``binary`` is ``False`` (default), the output is a dictionary\n    where elements of ``seq`` are stored in a list keyed to the value\n    of keyfunc for that element. If ``binary`` is True then a tuple\n    with lists ``T`` and ``F`` are returned where ``T`` is a list\n    containing elements of seq for which ``keyfunc`` was ``True`` and\n    ``F`` containing those elements for which ``keyfunc`` was ``False``;\n    a ValueError is raised if the ``keyfunc`` is not binary.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities import sift\n    >>> from sympy.abc import x, y\n    >>> from sympy import sqrt, exp, pi, Tuple\n\n    >>> sift(range(5), lambda x: x % 2)\n    {0: [0, 2, 4], 1: [1, 3]}\n\n    sift() returns a defaultdict() object, so any key that has no matches will\n    give [].\n\n    >>> sift([x], lambda x: x.is_commutative)\n    {True: [x]}\n    >>> _[False]\n    []\n\n    Sometimes you will not know how many keys you will get:\n\n    >>> sift([sqrt(x), exp(x), (y**x)**2],\n    ...      lambda x: x.as_base_exp()[0])\n    {E: [exp(x)], x: [sqrt(x)], y: [y**(2*x)]}\n\n    Sometimes you expect the results to be binary; the\n    results can be unpacked by setting ``binary`` to True:\n\n    >>> sift(range(4), lambda x: x % 2, binary=True)\n    ([1, 3], [0, 2])\n    >>> sift(Tuple(1, pi), lambda x: x.is_rational, binary=True)\n    ([1], [pi])\n\n    A ValueError is raised if the predicate was not actually binary\n    (which is a good test for the logic where sifting is used and\n    binary results were expected):\n\n    >>> unknown = exp(1) - pi  # the rationality of this is unknown\n    >>> args = Tuple(1, pi, unknown)\n    >>> sift(args, lambda x: x.is_rational, binary=True)\n    Traceback (most recent call last):\n    ...\n    ValueError: keyfunc gave non-binary output\n\n    The non-binary sifting shows that there were 3 keys generated:\n\n    >>> set(sift(args, lambda x: x.is_rational).keys())\n    {None, False, True}\n\n    If you need to sort the sifted items it might be better to use\n    ``ordered`` which can economically apply multiple sort keys\n    to a sequence while sorting.\n\n    See Also\n    ========\n\n    ordered\n\n    '
    if not binary:
        m = defaultdict(list)
        for i in seq:
            m[keyfunc(i)].append(i)
        return m
    sift = (F, T) = ([], [])
    for i in seq:
        try:
            sift[keyfunc(i)].append(i)
        except (IndexError, TypeError):
            raise ValueError('keyfunc gave non-binary output')
    return (T, F)

def take(iter, n):
    if False:
        print('Hello World!')
    'Return ``n`` items from ``iter`` iterator. '
    return [value for (_, value) in zip(range(n), iter)]

def dict_merge(*dicts):
    if False:
        return 10
    'Merge dictionaries into a single dictionary. '
    merged = {}
    for dict in dicts:
        merged.update(dict)
    return merged

def common_prefix(*seqs):
    if False:
        return 10
    'Return the subsequence that is a common start of sequences in ``seqs``.\n\n    >>> from sympy.utilities.iterables import common_prefix\n    >>> common_prefix(list(range(3)))\n    [0, 1, 2]\n    >>> common_prefix(list(range(3)), list(range(4)))\n    [0, 1, 2]\n    >>> common_prefix([1, 2, 3], [1, 2, 5])\n    [1, 2]\n    >>> common_prefix([1, 2, 3], [1, 3, 5])\n    [1]\n    '
    if not all(seqs):
        return []
    elif len(seqs) == 1:
        return seqs[0]
    i = 0
    for i in range(min((len(s) for s in seqs))):
        if not all((seqs[j][i] == seqs[0][i] for j in range(len(seqs)))):
            break
    else:
        i += 1
    return seqs[0][:i]

def common_suffix(*seqs):
    if False:
        print('Hello World!')
    'Return the subsequence that is a common ending of sequences in ``seqs``.\n\n    >>> from sympy.utilities.iterables import common_suffix\n    >>> common_suffix(list(range(3)))\n    [0, 1, 2]\n    >>> common_suffix(list(range(3)), list(range(4)))\n    []\n    >>> common_suffix([1, 2, 3], [9, 2, 3])\n    [2, 3]\n    >>> common_suffix([1, 2, 3], [9, 7, 3])\n    [3]\n    '
    if not all(seqs):
        return []
    elif len(seqs) == 1:
        return seqs[0]
    i = 0
    for i in range(-1, -min((len(s) for s in seqs)) - 1, -1):
        if not all((seqs[j][i] == seqs[0][i] for j in range(len(seqs)))):
            break
    else:
        i -= 1
    if i == -1:
        return []
    else:
        return seqs[0][i + 1:]

def prefixes(seq):
    if False:
        i = 10
        return i + 15
    '\n    Generate all prefixes of a sequence.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import prefixes\n\n    >>> list(prefixes([1,2,3,4]))\n    [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]]\n\n    '
    n = len(seq)
    for i in range(n):
        yield seq[:i + 1]

def postfixes(seq):
    if False:
        print('Hello World!')
    '\n    Generate all postfixes of a sequence.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import postfixes\n\n    >>> list(postfixes([1,2,3,4]))\n    [[4], [3, 4], [2, 3, 4], [1, 2, 3, 4]]\n\n    '
    n = len(seq)
    for i in range(n):
        yield seq[n - i - 1:]

def topological_sort(graph, key=None):
    if False:
        i = 10
        return i + 15
    "\n    Topological sort of graph's vertices.\n\n    Parameters\n    ==========\n\n    graph : tuple[list, list[tuple[T, T]]\n        A tuple consisting of a list of vertices and a list of edges of\n        a graph to be sorted topologically.\n\n    key : callable[T] (optional)\n        Ordering key for vertices on the same level. By default the natural\n        (e.g. lexicographic) ordering is used (in this case the base type\n        must implement ordering relations).\n\n    Examples\n    ========\n\n    Consider a graph::\n\n        +---+     +---+     +---+\n        | 7 |\\    | 5 |     | 3 |\n        +---+ \\   +---+     +---+\n          |   _\\___/ ____   _/ |\n          |  /  \\___/    \\ /   |\n          V  V           V V   |\n         +----+         +---+  |\n         | 11 |         | 8 |  |\n         +----+         +---+  |\n          | | \\____   ___/ _   |\n          | \\      \\ /    / \\  |\n          V  \\     V V   /  V  V\n        +---+ \\   +---+ |  +----+\n        | 2 |  |  | 9 | |  | 10 |\n        +---+  |  +---+ |  +----+\n               \\________/\n\n    where vertices are integers. This graph can be encoded using\n    elementary Python's data structures as follows::\n\n        >>> V = [2, 3, 5, 7, 8, 9, 10, 11]\n        >>> E = [(7, 11), (7, 8), (5, 11), (3, 8), (3, 10),\n        ...      (11, 2), (11, 9), (11, 10), (8, 9)]\n\n    To compute a topological sort for graph ``(V, E)`` issue::\n\n        >>> from sympy.utilities.iterables import topological_sort\n\n        >>> topological_sort((V, E))\n        [3, 5, 7, 8, 11, 2, 9, 10]\n\n    If specific tie breaking approach is needed, use ``key`` parameter::\n\n        >>> topological_sort((V, E), key=lambda v: -v)\n        [7, 5, 11, 3, 10, 8, 9, 2]\n\n    Only acyclic graphs can be sorted. If the input graph has a cycle,\n    then ``ValueError`` will be raised::\n\n        >>> topological_sort((V, E + [(10, 7)]))\n        Traceback (most recent call last):\n        ...\n        ValueError: cycle detected\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Topological_sorting\n\n    "
    (V, E) = graph
    L = []
    S = set(V)
    E = list(E)
    for (v, u) in E:
        S.discard(u)
    if key is None:

        def key(value):
            if False:
                for i in range(10):
                    print('nop')
            return value
    S = sorted(S, key=key, reverse=True)
    while S:
        node = S.pop()
        L.append(node)
        for (u, v) in list(E):
            if u == node:
                E.remove((u, v))
                for (_u, _v) in E:
                    if v == _v:
                        break
                else:
                    kv = key(v)
                    for (i, s) in enumerate(S):
                        ks = key(s)
                        if kv > ks:
                            S.insert(i, v)
                            break
                    else:
                        S.append(v)
    if E:
        raise ValueError('cycle detected')
    else:
        return L

def strongly_connected_components(G):
    if False:
        while True:
            i = 10
    "\n    Strongly connected components of a directed graph in reverse topological\n    order.\n\n\n    Parameters\n    ==========\n\n    G : tuple[list, list[tuple[T, T]]\n        A tuple consisting of a list of vertices and a list of edges of\n        a graph whose strongly connected components are to be found.\n\n\n    Examples\n    ========\n\n    Consider a directed graph (in dot notation)::\n\n        digraph {\n            A -> B\n            A -> C\n            B -> C\n            C -> B\n            B -> D\n        }\n\n    .. graphviz::\n\n        digraph {\n            A -> B\n            A -> C\n            B -> C\n            C -> B\n            B -> D\n        }\n\n    where vertices are the letters A, B, C and D. This graph can be encoded\n    using Python's elementary data structures as follows::\n\n        >>> V = ['A', 'B', 'C', 'D']\n        >>> E = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'B'), ('B', 'D')]\n\n    The strongly connected components of this graph can be computed as\n\n        >>> from sympy.utilities.iterables import strongly_connected_components\n\n        >>> strongly_connected_components((V, E))\n        [['D'], ['B', 'C'], ['A']]\n\n    This also gives the components in reverse topological order.\n\n    Since the subgraph containing B and C has a cycle they must be together in\n    a strongly connected component. A and D are connected to the rest of the\n    graph but not in a cyclic manner so they appear as their own strongly\n    connected components.\n\n\n    Notes\n    =====\n\n    The vertices of the graph must be hashable for the data structures used.\n    If the vertices are unhashable replace them with integer indices.\n\n    This function uses Tarjan's algorithm to compute the strongly connected\n    components in `O(|V|+|E|)` (linear) time.\n\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Strongly_connected_component\n    .. [2] https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm\n\n\n    See Also\n    ========\n\n    sympy.utilities.iterables.connected_components\n\n    "
    (V, E) = G
    Gmap = {vi: [] for vi in V}
    for (v1, v2) in E:
        Gmap[v1].append(v2)
    return _strongly_connected_components(V, Gmap)

def _strongly_connected_components(V, Gmap):
    if False:
        print('Hello World!')
    'More efficient internal routine for strongly_connected_components'
    lowlink = {}
    indices = {}
    stack = OrderedDict()
    callstack = []
    components = []
    nomore = object()

    def start(v):
        if False:
            return 10
        index = len(stack)
        indices[v] = lowlink[v] = index
        stack[v] = None
        callstack.append((v, iter(Gmap[v])))

    def finish(v1):
        if False:
            while True:
                i = 10
        if lowlink[v1] == indices[v1]:
            component = [stack.popitem()[0]]
            while component[-1] is not v1:
                component.append(stack.popitem()[0])
            components.append(component[::-1])
        (v2, _) = callstack.pop()
        if callstack:
            (v1, _) = callstack[-1]
            lowlink[v1] = min(lowlink[v1], lowlink[v2])
    for v in V:
        if v in indices:
            continue
        start(v)
        while callstack:
            (v1, it1) = callstack[-1]
            v2 = next(it1, nomore)
            if v2 is nomore:
                finish(v1)
            elif v2 not in indices:
                start(v2)
            elif v2 in stack:
                lowlink[v1] = min(lowlink[v1], indices[v2])
    return components

def connected_components(G):
    if False:
        print('Hello World!')
    "\n    Connected components of an undirected graph or weakly connected components\n    of a directed graph.\n\n\n    Parameters\n    ==========\n\n    G : tuple[list, list[tuple[T, T]]\n        A tuple consisting of a list of vertices and a list of edges of\n        a graph whose connected components are to be found.\n\n\n    Examples\n    ========\n\n\n    Given an undirected graph::\n\n        graph {\n            A -- B\n            C -- D\n        }\n\n    .. graphviz::\n\n        graph {\n            A -- B\n            C -- D\n        }\n\n    We can find the connected components using this function if we include\n    each edge in both directions::\n\n        >>> from sympy.utilities.iterables import connected_components\n\n        >>> V = ['A', 'B', 'C', 'D']\n        >>> E = [('A', 'B'), ('B', 'A'), ('C', 'D'), ('D', 'C')]\n        >>> connected_components((V, E))\n        [['A', 'B'], ['C', 'D']]\n\n    The weakly connected components of a directed graph can found the same\n    way.\n\n\n    Notes\n    =====\n\n    The vertices of the graph must be hashable for the data structures used.\n    If the vertices are unhashable replace them with integer indices.\n\n    This function uses Tarjan's algorithm to compute the connected components\n    in `O(|V|+|E|)` (linear) time.\n\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Component_%28graph_theory%29\n    .. [2] https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm\n\n\n    See Also\n    ========\n\n    sympy.utilities.iterables.strongly_connected_components\n\n    "
    (V, E) = G
    E_undirected = []
    for (v1, v2) in E:
        E_undirected.extend([(v1, v2), (v2, v1)])
    return strongly_connected_components((V, E_undirected))

def rotate_left(x, y):
    if False:
        print('Hello World!')
    '\n    Left rotates a list x by the number of steps specified\n    in y.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import rotate_left\n    >>> a = [0, 1, 2]\n    >>> rotate_left(a, 1)\n    [1, 2, 0]\n    '
    if len(x) == 0:
        return []
    y = y % len(x)
    return x[y:] + x[:y]

def rotate_right(x, y):
    if False:
        while True:
            i = 10
    '\n    Right rotates a list x by the number of steps specified\n    in y.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import rotate_right\n    >>> a = [0, 1, 2]\n    >>> rotate_right(a, 1)\n    [2, 0, 1]\n    '
    if len(x) == 0:
        return []
    y = len(x) - y % len(x)
    return x[y:] + x[:y]

def least_rotation(x, key=None):
    if False:
        i = 10
        return i + 15
    '\n    Returns the number of steps of left rotation required to\n    obtain lexicographically minimal string/list/tuple, etc.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import least_rotation, rotate_left\n    >>> a = [3, 1, 5, 1, 2]\n    >>> least_rotation(a)\n    3\n    >>> rotate_left(a, _)\n    [1, 2, 3, 1, 5]\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Lexicographically_minimal_string_rotation\n\n    '
    from sympy.functions.elementary.miscellaneous import Id
    if key is None:
        key = Id
    S = x + x
    f = [-1] * len(S)
    k = 0
    for j in range(1, len(S)):
        sj = S[j]
        i = f[j - k - 1]
        while i != -1 and sj != S[k + i + 1]:
            if key(sj) < key(S[k + i + 1]):
                k = j - i - 1
            i = f[i]
        if sj != S[k + i + 1]:
            if key(sj) < key(S[k]):
                k = j
            f[j - k] = -1
        else:
            f[j - k] = i + 1
    return k

def multiset_combinations(m, n, g=None):
    if False:
        print('Hello World!')
    "\n    Return the unique combinations of size ``n`` from multiset ``m``.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import multiset_combinations\n    >>> from itertools import combinations\n    >>> [''.join(i) for i in  multiset_combinations('baby', 3)]\n    ['abb', 'aby', 'bby']\n\n    >>> def count(f, s): return len(list(f(s, 3)))\n\n    The number of combinations depends on the number of letters; the\n    number of unique combinations depends on how the letters are\n    repeated.\n\n    >>> s1 = 'abracadabra'\n    >>> s2 = 'banana tree'\n    >>> count(combinations, s1), count(multiset_combinations, s1)\n    (165, 23)\n    >>> count(combinations, s2), count(multiset_combinations, s2)\n    (165, 54)\n\n    "
    from sympy.core.sorting import ordered
    if g is None:
        if isinstance(m, dict):
            if any((as_int(v) < 0 for v in m.values())):
                raise ValueError('counts cannot be negative')
            N = sum(m.values())
            if n > N:
                return
            g = [[k, m[k]] for k in ordered(m)]
        else:
            m = list(m)
            N = len(m)
            if n > N:
                return
            try:
                m = multiset(m)
                g = [(k, m[k]) for k in ordered(m)]
            except TypeError:
                m = list(ordered(m))
                g = [list(i) for i in group(m, multiple=False)]
        del m
    else:
        N = sum((v for (k, v) in g))
    if n > N or not n:
        yield []
    else:
        for (i, (k, v)) in enumerate(g):
            if v >= n:
                yield ([k] * n)
                v = n - 1
            for v in range(min(n, v), 0, -1):
                for j in multiset_combinations(None, n - v, g[i + 1:]):
                    rv = [k] * v + j
                    if len(rv) == n:
                        yield rv

def multiset_permutations(m, size=None, g=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the unique permutations of multiset ``m``.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import multiset_permutations\n    >>> from sympy import factorial\n    >>> [''.join(i) for i in multiset_permutations('aab')]\n    ['aab', 'aba', 'baa']\n    >>> factorial(len('banana'))\n    720\n    >>> len(list(multiset_permutations('banana')))\n    60\n    "
    from sympy.core.sorting import ordered
    if g is None:
        if isinstance(m, dict):
            if any((as_int(v) < 0 for v in m.values())):
                raise ValueError('counts cannot be negative')
            g = [[k, m[k]] for k in ordered(m)]
        else:
            m = list(ordered(m))
            g = [list(i) for i in group(m, multiple=False)]
        del m
    do = [gi for gi in g if gi[1] > 0]
    SUM = sum([gi[1] for gi in do])
    if not do or (size is not None and (size > SUM or size < 1)):
        if not do and size is None or size == 0:
            yield []
        return
    elif size == 1:
        for (k, v) in do:
            yield [k]
    elif len(do) == 1:
        (k, v) = do[0]
        v = v if size is None else size if size <= v else 0
        yield [k for i in range(v)]
    elif all((v == 1 for (k, v) in do)):
        for p in permutations([k for (k, v) in do], size):
            yield list(p)
    else:
        size = size if size is not None else SUM
        for (i, (k, v)) in enumerate(do):
            do[i][1] -= 1
            for j in multiset_permutations(None, size - 1, do):
                if j:
                    yield ([k] + j)
            do[i][1] += 1

def _partition(seq, vector, m=None):
    if False:
        i = 10
        return i + 15
    "\n    Return the partition of seq as specified by the partition vector.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import _partition\n    >>> _partition('abcde', [1, 0, 1, 2, 0])\n    [['b', 'e'], ['a', 'c'], ['d']]\n\n    Specifying the number of bins in the partition is optional:\n\n    >>> _partition('abcde', [1, 0, 1, 2, 0], 3)\n    [['b', 'e'], ['a', 'c'], ['d']]\n\n    The output of _set_partitions can be passed as follows:\n\n    >>> output = (3, [1, 0, 1, 2, 0])\n    >>> _partition('abcde', *output)\n    [['b', 'e'], ['a', 'c'], ['d']]\n\n    See Also\n    ========\n\n    combinatorics.partitions.Partition.from_rgs\n\n    "
    if m is None:
        m = max(vector) + 1
    elif isinstance(vector, int):
        (vector, m) = (m, vector)
    p = [[] for i in range(m)]
    for (i, v) in enumerate(vector):
        p[v].append(seq[i])
    return p

def _set_partitions(n):
    if False:
        for i in range(10):
            print('nop')
    'Cycle through all partitions of n elements, yielding the\n    current number of partitions, ``m``, and a mutable list, ``q``\n    such that ``element[i]`` is in part ``q[i]`` of the partition.\n\n    NOTE: ``q`` is modified in place and generally should not be changed\n    between function calls.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import _set_partitions, _partition\n    >>> for m, q in _set_partitions(3):\n    ...     print(\'%s %s %s\' % (m, q, _partition(\'abc\', q, m)))\n    1 [0, 0, 0] [[\'a\', \'b\', \'c\']]\n    2 [0, 0, 1] [[\'a\', \'b\'], [\'c\']]\n    2 [0, 1, 0] [[\'a\', \'c\'], [\'b\']]\n    2 [0, 1, 1] [[\'a\'], [\'b\', \'c\']]\n    3 [0, 1, 2] [[\'a\'], [\'b\'], [\'c\']]\n\n    Notes\n    =====\n\n    This algorithm is similar to, and solves the same problem as,\n    Algorithm 7.2.1.5H, from volume 4A of Knuth\'s The Art of Computer\n    Programming.  Knuth uses the term "restricted growth string" where\n    this code refers to a "partition vector". In each case, the meaning is\n    the same: the value in the ith element of the vector specifies to\n    which part the ith set element is to be assigned.\n\n    At the lowest level, this code implements an n-digit big-endian\n    counter (stored in the array q) which is incremented (with carries) to\n    get the next partition in the sequence.  A special twist is that a\n    digit is constrained to be at most one greater than the maximum of all\n    the digits to the left of it.  The array p maintains this maximum, so\n    that the code can efficiently decide when a digit can be incremented\n    in place or whether it needs to be reset to 0 and trigger a carry to\n    the next digit.  The enumeration starts with all the digits 0 (which\n    corresponds to all the set elements being assigned to the same 0th\n    part), and ends with 0123...n, which corresponds to each set element\n    being assigned to a different, singleton, part.\n\n    This routine was rewritten to use 0-based lists while trying to\n    preserve the beauty and efficiency of the original algorithm.\n\n    References\n    ==========\n\n    .. [1] Nijenhuis, Albert and Wilf, Herbert. (1978) Combinatorial Algorithms,\n        2nd Ed, p 91, algorithm "nexequ". Available online from\n        https://www.math.upenn.edu/~wilf/website/CombAlgDownld.html (viewed\n        November 17, 2012).\n\n    '
    p = [0] * n
    q = [0] * n
    nc = 1
    yield (nc, q)
    while nc != n:
        m = n
        while 1:
            m -= 1
            i = q[m]
            if p[i] != 1:
                break
            q[m] = 0
        i += 1
        q[m] = i
        m += 1
        nc += m - n
        p[0] += n - m
        if i == nc:
            p[nc] = 0
            nc += 1
        p[i - 1] -= 1
        p[i] += 1
        yield (nc, q)

def multiset_partitions(multiset, m=None):
    if False:
        print('Hello World!')
    '\n    Return unique partitions of the given multiset (in list form).\n    If ``m`` is None, all multisets will be returned, otherwise only\n    partitions with ``m`` parts will be returned.\n\n    If ``multiset`` is an integer, a range [0, 1, ..., multiset - 1]\n    will be supplied.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import multiset_partitions\n    >>> list(multiset_partitions([1, 2, 3, 4], 2))\n    [[[1, 2, 3], [4]], [[1, 2, 4], [3]], [[1, 2], [3, 4]],\n    [[1, 3, 4], [2]], [[1, 3], [2, 4]], [[1, 4], [2, 3]],\n    [[1], [2, 3, 4]]]\n    >>> list(multiset_partitions([1, 2, 3, 4], 1))\n    [[[1, 2, 3, 4]]]\n\n    Only unique partitions are returned and these will be returned in a\n    canonical order regardless of the order of the input:\n\n    >>> a = [1, 2, 2, 1]\n    >>> ans = list(multiset_partitions(a, 2))\n    >>> a.sort()\n    >>> list(multiset_partitions(a, 2)) == ans\n    True\n    >>> a = range(3, 1, -1)\n    >>> (list(multiset_partitions(a)) ==\n    ...  list(multiset_partitions(sorted(a))))\n    True\n\n    If m is omitted then all partitions will be returned:\n\n    >>> list(multiset_partitions([1, 1, 2]))\n    [[[1, 1, 2]], [[1, 1], [2]], [[1, 2], [1]], [[1], [1], [2]]]\n    >>> list(multiset_partitions([1]*3))\n    [[[1, 1, 1]], [[1], [1, 1]], [[1], [1], [1]]]\n\n    Counting\n    ========\n\n    The number of partitions of a set is given by the bell number:\n\n    >>> from sympy import bell\n    >>> len(list(multiset_partitions(5))) == bell(5) == 52\n    True\n\n    The number of partitions of length k from a set of size n is given by the\n    Stirling Number of the 2nd kind:\n\n    >>> from sympy.functions.combinatorial.numbers import stirling\n    >>> stirling(5, 2) == len(list(multiset_partitions(5, 2))) == 15\n    True\n\n    These comments on counting apply to *sets*, not multisets.\n\n    Notes\n    =====\n\n    When all the elements are the same in the multiset, the order\n    of the returned partitions is determined by the ``partitions``\n    routine. If one is counting partitions then it is better to use\n    the ``nT`` function.\n\n    See Also\n    ========\n\n    partitions\n    sympy.combinatorics.partitions.Partition\n    sympy.combinatorics.partitions.IntegerPartition\n    sympy.functions.combinatorial.numbers.nT\n\n    '
    if isinstance(multiset, int):
        n = multiset
        if m and m > n:
            return
        multiset = list(range(n))
        if m == 1:
            yield [multiset[:]]
            return
        for (nc, q) in _set_partitions(n):
            if m is None or nc == m:
                rv = [[] for i in range(nc)]
                for i in range(n):
                    rv[q[i]].append(multiset[i])
                yield rv
        return
    if len(multiset) == 1 and isinstance(multiset, str):
        multiset = [multiset]
    if not has_variety(multiset):
        n = len(multiset)
        if m and m > n:
            return
        if m == 1:
            yield [multiset[:]]
            return
        x = multiset[:1]
        for (size, p) in partitions(n, m, size=True):
            if m is None or size == m:
                rv = []
                for k in sorted(p):
                    rv.extend([x * k] * p[k])
                yield rv
    else:
        from sympy.core.sorting import ordered
        multiset = list(ordered(multiset))
        n = len(multiset)
        if m and m > n:
            return
        if m == 1:
            yield [multiset[:]]
            return
        (elements, multiplicities) = zip(*group(multiset, False))
        if len(elements) < len(multiset):
            if m:
                mpt = MultisetPartitionTraverser()
                for state in mpt.enum_range(multiplicities, m - 1, m):
                    yield list_visitor(state, elements)
            else:
                for state in multiset_partitions_taocp(multiplicities):
                    yield list_visitor(state, elements)
        else:
            for (nc, q) in _set_partitions(n):
                if m is None or nc == m:
                    rv = [[] for i in range(nc)]
                    for i in range(n):
                        rv[q[i]].append(i)
                    yield [[multiset[j] for j in i] for i in rv]

def partitions(n, m=None, k=None, size=False):
    if False:
        print('Hello World!')
    'Generate all partitions of positive integer, n.\n\n    Each partition is represented as a dictionary, mapping an integer\n    to the number of copies of that integer in the partition.  For example,\n    the first partition of 4 returned is {4: 1}, "4: one of them".\n\n    Parameters\n    ==========\n    n : int\n    m : int, optional\n        limits number of parts in partition (mnemonic: m, maximum parts)\n    k : int, optional\n        limits the numbers that are kept in the partition (mnemonic: k, keys)\n    size : bool, default: False\n        If ``True``, (M, P) is returned where M is the sum of the\n        multiplicities and P is the generated partition.\n        If ``False``, only the generated partition is returned.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import partitions\n\n    The numbers appearing in the partition (the key of the returned dict)\n    are limited with k:\n\n    >>> for p in partitions(6, k=2):  # doctest: +SKIP\n    ...     print(p)\n    {2: 3}\n    {1: 2, 2: 2}\n    {1: 4, 2: 1}\n    {1: 6}\n\n    The maximum number of parts in the partition (the sum of the values in\n    the returned dict) are limited with m (default value, None, gives\n    partitions from 1 through n):\n\n    >>> for p in partitions(6, m=2):  # doctest: +SKIP\n    ...     print(p)\n    ...\n    {6: 1}\n    {1: 1, 5: 1}\n    {2: 1, 4: 1}\n    {3: 2}\n\n    References\n    ==========\n\n    .. [1] modified from Tim Peter\'s version to allow for k and m values:\n           https://code.activestate.com/recipes/218332-generator-for-integer-partitions/\n\n    See Also\n    ========\n\n    sympy.combinatorics.partitions.Partition\n    sympy.combinatorics.partitions.IntegerPartition\n\n    '
    if n <= 0 or (m is not None and m < 1) or (k is not None and k < 1) or (m and k and (m * k < n)):
        if size:
            yield (0, {})
        else:
            yield {}
        return
    if m is None:
        m = n
    else:
        m = min(m, n)
    k = min(k or n, n)
    (n, m, k) = (as_int(n), as_int(m), as_int(k))
    (q, r) = divmod(n, k)
    ms = {k: q}
    keys = [k]
    if r:
        ms[r] = 1
        keys.append(r)
    room = m - q - bool(r)
    if size:
        yield (sum(ms.values()), ms.copy())
    else:
        yield ms.copy()
    while keys != [1]:
        if keys[-1] == 1:
            del keys[-1]
            reuse = ms.pop(1)
            room += reuse
        else:
            reuse = 0
        while 1:
            i = keys[-1]
            newcount = ms[i] = ms[i] - 1
            reuse += i
            if newcount == 0:
                del keys[-1], ms[i]
            room += 1
            i -= 1
            (q, r) = divmod(reuse, i)
            need = q + bool(r)
            if need > room:
                if not keys:
                    return
                continue
            ms[i] = q
            keys.append(i)
            if r:
                ms[r] = 1
                keys.append(r)
            break
        room -= need
        if size:
            yield (sum(ms.values()), ms.copy())
        else:
            yield ms.copy()

def ordered_partitions(n, m=None, sort=True):
    if False:
        i = 10
        return i + 15
    'Generates ordered partitions of integer *n*.\n\n    Parameters\n    ==========\n    n : int\n    m : int, optional\n        The default value gives partitions of all sizes else only\n        those with size m. In addition, if *m* is not None then\n        partitions are generated *in place* (see examples).\n    sort : bool, default: True\n        Controls whether partitions are\n        returned in sorted order when *m* is not None; when False,\n        the partitions are returned as fast as possible with elements\n        sorted, but when m|n the partitions will not be in\n        ascending lexicographical order.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import ordered_partitions\n\n    All partitions of 5 in ascending lexicographical:\n\n    >>> for p in ordered_partitions(5):\n    ...     print(p)\n    [1, 1, 1, 1, 1]\n    [1, 1, 1, 2]\n    [1, 1, 3]\n    [1, 2, 2]\n    [1, 4]\n    [2, 3]\n    [5]\n\n    Only partitions of 5 with two parts:\n\n    >>> for p in ordered_partitions(5, 2):\n    ...     print(p)\n    [1, 4]\n    [2, 3]\n\n    When ``m`` is given, a given list objects will be used more than\n    once for speed reasons so you will not see the correct partitions\n    unless you make a copy of each as it is generated:\n\n    >>> [p for p in ordered_partitions(7, 3)]\n    [[1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 2, 2]]\n    >>> [list(p) for p in ordered_partitions(7, 3)]\n    [[1, 1, 5], [1, 2, 4], [1, 3, 3], [2, 2, 3]]\n\n    When ``n`` is a multiple of ``m``, the elements are still sorted\n    but the partitions themselves will be *unordered* if sort is False;\n    the default is to return them in ascending lexicographical order.\n\n    >>> for p in ordered_partitions(6, 2):\n    ...     print(p)\n    [1, 5]\n    [2, 4]\n    [3, 3]\n\n    But if speed is more important than ordering, sort can be set to\n    False:\n\n    >>> for p in ordered_partitions(6, 2, sort=False):\n    ...     print(p)\n    [1, 5]\n    [3, 3]\n    [2, 4]\n\n    References\n    ==========\n\n    .. [1] Generating Integer Partitions, [online],\n        Available: https://jeromekelleher.net/generating-integer-partitions.html\n    .. [2] Jerome Kelleher and Barry O\'Sullivan, "Generating All\n        Partitions: A Comparison Of Two Encodings", [online],\n        Available: https://arxiv.org/pdf/0909.2331v2.pdf\n    '
    if n < 1 or (m is not None and m < 1):
        yield []
        return
    if m is None:
        a = [1] * n
        y = -1
        v = n
        while v > 0:
            v -= 1
            x = a[v] + 1
            while y >= 2 * x:
                a[v] = x
                y -= x
                v += 1
            w = v + 1
            while x <= y:
                a[v] = x
                a[w] = y
                yield a[:w + 1]
                x += 1
                y -= 1
            a[v] = x + y
            y = a[v] - 1
            yield a[:w]
    elif m == 1:
        yield [n]
    elif n == m:
        yield ([1] * n)
    else:
        for b in range(1, n // m + 1):
            a = [b] * m
            x = n - b * m
            if not x:
                if sort:
                    yield a
            elif not sort and x <= m:
                for ax in ordered_partitions(x, sort=False):
                    mi = len(ax)
                    a[-mi:] = [i + b for i in ax]
                    yield a
                    a[-mi:] = [b] * mi
            else:
                for mi in range(1, m):
                    for ax in ordered_partitions(x, mi, sort=True):
                        a[-mi:] = [i + b for i in ax]
                        yield a
                        a[-mi:] = [b] * mi

def binary_partitions(n):
    if False:
        i = 10
        return i + 15
    '\n    Generates the binary partition of *n*.\n\n    A binary partition consists only of numbers that are\n    powers of two. Each step reduces a `2^{k+1}` to `2^k` and\n    `2^k`. Thus 16 is converted to 8 and 8.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import binary_partitions\n    >>> for i in binary_partitions(5):\n    ...     print(i)\n    ...\n    [4, 1]\n    [2, 2, 1]\n    [2, 1, 1, 1]\n    [1, 1, 1, 1, 1]\n\n    References\n    ==========\n\n    .. [1] TAOCP 4, section 7.2.1.5, problem 64\n\n    '
    from math import ceil, log
    power = int(2 ** ceil(log(n, 2)))
    acc = 0
    partition = []
    while power:
        if acc + power <= n:
            partition.append(power)
            acc += power
        power >>= 1
    last_num = len(partition) - 1 - (n & 1)
    while last_num >= 0:
        yield partition
        if partition[last_num] == 2:
            partition[last_num] = 1
            partition.append(1)
            last_num -= 1
            continue
        partition.append(1)
        partition[last_num] >>= 1
        x = partition[last_num + 1] = partition[last_num]
        last_num += 1
        while x > 1:
            if x <= len(partition) - last_num - 1:
                del partition[-x + 1:]
                last_num += 1
                partition[last_num] = x
            else:
                x >>= 1
    yield ([1] * n)

def has_dups(seq):
    if False:
        return 10
    'Return True if there are any duplicate elements in ``seq``.\n\n    Examples\n    ========\n\n    >>> from sympy import has_dups, Dict, Set\n    >>> has_dups((1, 2, 1))\n    True\n    >>> has_dups(range(3))\n    False\n    >>> all(has_dups(c) is False for c in (set(), Set(), dict(), Dict()))\n    True\n    '
    from sympy.core.containers import Dict
    from sympy.sets.sets import Set
    if isinstance(seq, (dict, set, Dict, Set)):
        return False
    unique = set()
    try:
        return any((True for s in seq if s in unique or unique.add(s)))
    except TypeError:
        return len(seq) != len(list(uniq(seq)))

def has_variety(seq):
    if False:
        for i in range(10):
            print('nop')
    'Return True if there are any different elements in ``seq``.\n\n    Examples\n    ========\n\n    >>> from sympy import has_variety\n\n    >>> has_variety((1, 2, 1))\n    True\n    >>> has_variety((1, 1, 1))\n    False\n    '
    for (i, s) in enumerate(seq):
        if i == 0:
            sentinel = s
        elif s != sentinel:
            return True
    return False

def uniq(seq, result=None):
    if False:
        return 10
    '\n    Yield unique elements from ``seq`` as an iterator. The second\n    parameter ``result``  is used internally; it is not necessary\n    to pass anything for this.\n\n    Note: changing the sequence during iteration will raise a\n    RuntimeError if the size of the sequence is known; if you pass\n    an iterator and advance the iterator you will change the\n    output of this routine but there will be no warning.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import uniq\n    >>> dat = [1, 4, 1, 5, 4, 2, 1, 2]\n    >>> type(uniq(dat)) in (list, tuple)\n    False\n\n    >>> list(uniq(dat))\n    [1, 4, 5, 2]\n    >>> list(uniq(x for x in dat))\n    [1, 4, 5, 2]\n    >>> list(uniq([[1], [2, 1], [1]]))\n    [[1], [2, 1]]\n    '
    try:
        n = len(seq)
    except TypeError:
        n = None

    def check():
        if False:
            print('Hello World!')
        if n is not None and len(seq) != n:
            raise RuntimeError('sequence changed size during iteration')
    try:
        seen = set()
        result = result or []
        for (i, s) in enumerate(seq):
            if not (s in seen or seen.add(s)):
                yield s
                check()
    except TypeError:
        if s not in result:
            yield s
            check()
            result.append(s)
        if hasattr(seq, '__getitem__'):
            yield from uniq(seq[i + 1:], result)
        else:
            yield from uniq(seq, result)

def generate_bell(n):
    if False:
        while True:
            i = 10
    'Return permutations of [0, 1, ..., n - 1] such that each permutation\n    differs from the last by the exchange of a single pair of neighbors.\n    The ``n!`` permutations are returned as an iterator. In order to obtain\n    the next permutation from a random starting permutation, use the\n    ``next_trotterjohnson`` method of the Permutation class (which generates\n    the same sequence in a different manner).\n\n    Examples\n    ========\n\n    >>> from itertools import permutations\n    >>> from sympy.utilities.iterables import generate_bell\n    >>> from sympy import zeros, Matrix\n\n    This is the sort of permutation used in the ringing of physical bells,\n    and does not produce permutations in lexicographical order. Rather, the\n    permutations differ from each other by exactly one inversion, and the\n    position at which the swapping occurs varies periodically in a simple\n    fashion. Consider the first few permutations of 4 elements generated\n    by ``permutations`` and ``generate_bell``:\n\n    >>> list(permutations(range(4)))[:5]\n    [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1), (0, 3, 1, 2)]\n    >>> list(generate_bell(4))[:5]\n    [(0, 1, 2, 3), (0, 1, 3, 2), (0, 3, 1, 2), (3, 0, 1, 2), (3, 0, 2, 1)]\n\n    Notice how the 2nd and 3rd lexicographical permutations have 3 elements\n    out of place whereas each "bell" permutation always has only two\n    elements out of place relative to the previous permutation (and so the\n    signature (+/-1) of a permutation is opposite of the signature of the\n    previous permutation).\n\n    How the position of inversion varies across the elements can be seen\n    by tracing out where the largest number appears in the permutations:\n\n    >>> m = zeros(4, 24)\n    >>> for i, p in enumerate(generate_bell(4)):\n    ...     m[:, i] = Matrix([j - 3 for j in list(p)])  # make largest zero\n    >>> m.print_nonzero(\'X\')\n    [XXX  XXXXXX  XXXXXX  XXX]\n    [XX XX XXXX XX XXXX XX XX]\n    [X XXXX XX XXXX XX XXXX X]\n    [ XXXXXX  XXXXXX  XXXXXX ]\n\n    See Also\n    ========\n\n    sympy.combinatorics.permutations.Permutation.next_trotterjohnson\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Method_ringing\n\n    .. [2] https://stackoverflow.com/questions/4856615/recursive-permutation/4857018\n\n    .. [3] https://web.archive.org/web/20160313023044/http://programminggeeks.com/bell-algorithm-for-permutation/\n\n    .. [4] https://en.wikipedia.org/wiki/Steinhaus%E2%80%93Johnson%E2%80%93Trotter_algorithm\n\n    .. [5] Generating involutions, derangements, and relatives by ECO\n           Vincent Vajnovszki, DMTCS vol 1 issue 12, 2010\n\n    '
    n = as_int(n)
    if n < 1:
        raise ValueError('n must be a positive integer')
    if n == 1:
        yield (0,)
    elif n == 2:
        yield (0, 1)
        yield (1, 0)
    elif n == 3:
        yield from [(0, 1, 2), (0, 2, 1), (2, 0, 1), (2, 1, 0), (1, 2, 0), (1, 0, 2)]
    else:
        m = n - 1
        op = [0] + [-1] * m
        l = list(range(n))
        while True:
            yield tuple(l)
            big = (None, -1)
            for i in range(n):
                if op[i] and l[i] > big[1]:
                    big = (i, l[i])
            (i, _) = big
            if i is None:
                break
            j = i + op[i]
            (l[i], l[j]) = (l[j], l[i])
            (op[i], op[j]) = (op[j], op[i])
            if j == 0 or j == m or l[j + op[j]] > l[j]:
                op[j] = 0
            for i in range(j):
                if l[i] > l[j]:
                    op[i] = 1
            for i in range(j + 1, n):
                if l[i] > l[j]:
                    op[i] = -1

def generate_involutions(n):
    if False:
        print('Hello World!')
    '\n    Generates involutions.\n\n    An involution is a permutation that when multiplied\n    by itself equals the identity permutation. In this\n    implementation the involutions are generated using\n    Fixed Points.\n\n    Alternatively, an involution can be considered as\n    a permutation that does not contain any cycles with\n    a length that is greater than two.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import generate_involutions\n    >>> list(generate_involutions(3))\n    [(0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 1, 0)]\n    >>> len(list(generate_involutions(4)))\n    10\n\n    References\n    ==========\n\n    .. [1] https://mathworld.wolfram.com/PermutationInvolution.html\n\n    '
    idx = list(range(n))
    for p in permutations(idx):
        for i in idx:
            if p[p[i]] != i:
                break
        else:
            yield p

def multiset_derangements(s):
    if False:
        print('Hello World!')
    "Generate derangements of the elements of s *in place*.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import multiset_derangements, uniq\n\n    Because the derangements of multisets (not sets) are generated\n    in place, copies of the return value must be made if a collection\n    of derangements is desired or else all values will be the same:\n\n    >>> list(uniq([i for i in multiset_derangements('1233')]))\n    [[None, None, None, None]]\n    >>> [i.copy() for i in multiset_derangements('1233')]\n    [['3', '3', '1', '2'], ['3', '3', '2', '1']]\n    >>> [''.join(i) for i in multiset_derangements('1233')]\n    ['3312', '3321']\n    "
    from sympy.core.sorting import ordered
    try:
        ms = multiset(s)
    except TypeError:
        key = dict(enumerate(ordered(uniq(s))))
        h = []
        for si in s:
            for k in key:
                if key[k] == si:
                    h.append(k)
                    break
        for i in multiset_derangements(h):
            yield [key[j] for j in i]
        return
    mx = max(ms.values())
    n = len(s)
    if mx * 2 > n:
        return
    if len(ms) == n:
        yield from _set_derangements(s)
        return
    for M in ms:
        if ms[M] == mx:
            break
    inonM = [i for i in range(n) if s[i] != M]
    iM = [i for i in range(n) if s[i] == M]
    rv = [None] * n
    if 2 * mx == n:
        for i in inonM:
            rv[i] = M
        for p in multiset_permutations([s[i] for i in inonM]):
            for (i, pi) in zip(iM, p):
                rv[i] = pi
            yield rv
        rv[:] = [None] * n
        return
    if n - 2 * mx == 1 and len(ms.values()) == n - mx + 1:
        for (i, i1) in enumerate(inonM):
            ifill = inonM[:i] + inonM[i + 1:]
            for j in ifill:
                rv[j] = M
            for p in permutations([s[j] for j in ifill]):
                rv[i1] = s[i1]
                for (j, pi) in zip(iM, p):
                    rv[j] = pi
                k = i1
                for j in iM:
                    (rv[j], rv[k]) = (rv[k], rv[j])
                    yield rv
                    k = j
        rv[:] = [None] * n
        return

    def finish_derangements():
        if False:
            while True:
                i = 10
        'Place the last two elements into the partially completed\n        derangement, and yield the results.\n        '
        a = take[1][0]
        a_ct = take[1][1]
        b = take[0][0]
        b_ct = take[0][1]
        forced_a = []
        forced_b = []
        open_free = []
        for i in range(len(s)):
            if rv[i] is None:
                if s[i] == a:
                    forced_b.append(i)
                elif s[i] == b:
                    forced_a.append(i)
                else:
                    open_free.append(i)
        if len(forced_a) > a_ct or len(forced_b) > b_ct:
            return
        for i in forced_a:
            rv[i] = a
        for i in forced_b:
            rv[i] = b
        for a_place in combinations(open_free, a_ct - len(forced_a)):
            for a_pos in a_place:
                rv[a_pos] = a
            for i in open_free:
                if rv[i] is None:
                    rv[i] = b
            yield rv
            for i in open_free:
                rv[i] = None
        for i in forced_a:
            rv[i] = None
        for i in forced_b:
            rv[i] = None

    def iopen(v):
        if False:
            i = 10
            return i + 15
        return [i for i in range(n) if rv[i] is None and s[i] != v]

    def do(j):
        if False:
            i = 10
            return i + 15
        if j == 1:
            yield from finish_derangements()
        else:
            (M, mx) = take[j]
            for i in combinations(iopen(M), mx):
                for ii in i:
                    rv[ii] = M
                yield from do(j - 1)
                for ii in i:
                    rv[ii] = None
    take = sorted(ms.items(), key=lambda x: (x[1], x[0]))
    yield from do(len(take) - 1)
    rv[:] = [None] * n

def random_derangement(t, choice=None, strict=True):
    if False:
        print('Hello World!')
    "Return a list of elements in which none are in the same positions\n    as they were originally. If an element fills more than half of the positions\n    then an error will be raised since no derangement is possible. To obtain\n    a derangement of as many items as possible--with some of the most numerous\n    remaining in their original positions--pass `strict=False`. To produce a\n    pseudorandom derangment, pass a pseudorandom selector like `choice` (see\n    below).\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import random_derangement\n    >>> t = 'SymPy: a CAS in pure Python'\n    >>> d = random_derangement(t)\n    >>> all(i != j for i, j in zip(d, t))\n    True\n\n    A predictable result can be obtained by using a pseudorandom\n    generator for the choice:\n\n    >>> from sympy.core.random import seed, choice as c\n    >>> seed(1)\n    >>> d = [''.join(random_derangement(t, c)) for i in range(5)]\n    >>> assert len(set(d)) != 1  # we got different values\n\n    By reseeding, the same sequence can be obtained:\n\n    >>> seed(1)\n    >>> d2 = [''.join(random_derangement(t, c)) for i in range(5)]\n    >>> assert d == d2\n    "
    if choice is None:
        import secrets
        choice = secrets.choice

    def shuffle(rv):
        if False:
            i = 10
            return i + 15
        'Knuth shuffle'
        for i in range(len(rv) - 1, 0, -1):
            x = choice(rv[:i + 1])
            j = rv.index(x)
            (rv[i], rv[j]) = (rv[j], rv[i])

    def pick(rv, n):
        if False:
            print('Hello World!')
        'shuffle rv and return the first n values\n        '
        shuffle(rv)
        return rv[:n]
    ms = multiset(t)
    tot = len(t)
    ms = sorted(ms.items(), key=lambda x: x[1])
    (M, mx) = ms[-1]
    n = len(t)
    xs = 2 * mx - tot
    if xs > 0:
        if strict:
            raise ValueError('no derangement possible')
        opts = [i for (i, c) in enumerate(t) if c == ms[-1][0]]
        pick(opts, xs)
        stay = sorted(opts[:xs])
        rv = list(t)
        for i in reversed(stay):
            rv.pop(i)
        rv = random_derangement(rv, choice)
        for i in stay:
            rv.insert(i, ms[-1][0])
        return ''.join(rv) if type(t) is str else rv
    if n == len(ms):
        rv = list(t)
        while True:
            shuffle(rv)
            if all((i != j for (i, j) in zip(rv, t))):
                break
    else:
        rv = [None] * n
        while True:
            j = 0
            while j > -len(ms):
                j -= 1
                (e, c) = ms[j]
                opts = [i for i in range(n) if rv[i] is None and t[i] != e]
                if len(opts) < c:
                    for i in range(n):
                        rv[i] = None
                    break
                pick(opts, c)
                for i in range(c):
                    rv[opts[i]] = e
            else:
                return rv
    return rv

def _set_derangements(s):
    if False:
        print('Hello World!')
    '\n    yield derangements of items in ``s`` which are assumed to contain\n    no repeated elements\n    '
    if len(s) < 2:
        return
    if len(s) == 2:
        yield [s[1], s[0]]
        return
    if len(s) == 3:
        yield [s[1], s[2], s[0]]
        yield [s[2], s[0], s[1]]
        return
    for p in permutations(s):
        if not any((i == j for (i, j) in zip(p, s))):
            yield list(p)

def generate_derangements(s):
    if False:
        while True:
            i = 10
    '\n    Return unique derangements of the elements of iterable ``s``.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import generate_derangements\n    >>> list(generate_derangements([0, 1, 2]))\n    [[1, 2, 0], [2, 0, 1]]\n    >>> list(generate_derangements([0, 1, 2, 2]))\n    [[2, 2, 0, 1], [2, 2, 1, 0]]\n    >>> list(generate_derangements([0, 1, 1]))\n    []\n\n    See Also\n    ========\n\n    sympy.functions.combinatorial.factorials.subfactorial\n\n    '
    if not has_dups(s):
        yield from _set_derangements(s)
    else:
        for p in multiset_derangements(s):
            yield list(p)

def necklaces(n, k, free=False):
    if False:
        print('Hello World!')
    '\n    A routine to generate necklaces that may (free=True) or may not\n    (free=False) be turned over to be viewed. The "necklaces" returned\n    are comprised of ``n`` integers (beads) with ``k`` different\n    values (colors). Only unique necklaces are returned.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import necklaces, bracelets\n    >>> def show(s, i):\n    ...     return \'\'.join(s[j] for j in i)\n\n    The "unrestricted necklace" is sometimes also referred to as a\n    "bracelet" (an object that can be turned over, a sequence that can\n    be reversed) and the term "necklace" is used to imply a sequence\n    that cannot be reversed. So ACB == ABC for a bracelet (rotate and\n    reverse) while the two are different for a necklace since rotation\n    alone cannot make the two sequences the same.\n\n    (mnemonic: Bracelets can be viewed Backwards, but Not Necklaces.)\n\n    >>> B = [show(\'ABC\', i) for i in bracelets(3, 3)]\n    >>> N = [show(\'ABC\', i) for i in necklaces(3, 3)]\n    >>> set(N) - set(B)\n    {\'ACB\'}\n\n    >>> list(necklaces(4, 2))\n    [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 1),\n     (0, 1, 0, 1), (0, 1, 1, 1), (1, 1, 1, 1)]\n\n    >>> [show(\'.o\', i) for i in bracelets(4, 2)]\n    [\'....\', \'...o\', \'..oo\', \'.o.o\', \'.ooo\', \'oooo\']\n\n    References\n    ==========\n\n    .. [1] https://mathworld.wolfram.com/Necklace.html\n\n    .. [2] Frank Ruskey, Carla Savage, and Terry Min Yih Wang,\n        Generating necklaces, Journal of Algorithms 13 (1992), 414-430;\n        https://doi.org/10.1016/0196-6774(92)90047-G\n\n    '
    if k == 0 and n > 0:
        return
    a = [0] * n
    yield tuple(a)
    if n == 0:
        return
    while True:
        i = n - 1
        while a[i] == k - 1:
            i -= 1
            if i == -1:
                return
        a[i] += 1
        for j in range(n - i - 1):
            a[j + i + 1] = a[j]
        if n % (i + 1) == 0 and (not free or all((a <= a[j::-1] + a[-1:j:-1] for j in range(n - 1)))):
            yield tuple(a)

def bracelets(n, k):
    if False:
        print('Hello World!')
    'Wrapper to necklaces to return a free (unrestricted) necklace.'
    return necklaces(n, k, free=True)

def generate_oriented_forest(n):
    if False:
        return 10
    '\n    This algorithm generates oriented forests.\n\n    An oriented graph is a directed graph having no symmetric pair of directed\n    edges. A forest is an acyclic graph, i.e., it has no cycles. A forest can\n    also be described as a disjoint union of trees, which are graphs in which\n    any two vertices are connected by exactly one simple path.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import generate_oriented_forest\n    >>> list(generate_oriented_forest(4))\n    [[0, 1, 2, 3], [0, 1, 2, 2], [0, 1, 2, 1], [0, 1, 2, 0],     [0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0]]\n\n    References\n    ==========\n\n    .. [1] T. Beyer and S.M. Hedetniemi: constant time generation of\n           rooted trees, SIAM J. Computing Vol. 9, No. 4, November 1980\n\n    .. [2] https://stackoverflow.com/questions/1633833/oriented-forest-taocp-algorithm-in-python\n\n    '
    P = list(range(-1, n))
    while True:
        yield P[1:]
        if P[n] > 0:
            P[n] = P[P[n]]
        else:
            for p in range(n - 1, 0, -1):
                if P[p] != 0:
                    target = P[p] - 1
                    for q in range(p - 1, 0, -1):
                        if P[q] == target:
                            break
                    offset = p - q
                    for i in range(p, n + 1):
                        P[i] = P[i - offset]
                    break
            else:
                break

def minlex(seq, directed=True, key=None):
    if False:
        i = 10
        return i + 15
    "\n    Return the rotation of the sequence in which the lexically smallest\n    elements appear first, e.g. `cba \\rightarrow acb`.\n\n    The sequence returned is a tuple, unless the input sequence is a string\n    in which case a string is returned.\n\n    If ``directed`` is False then the smaller of the sequence and the\n    reversed sequence is returned, e.g. `cba \\rightarrow abc`.\n\n    If ``key`` is not None then it is used to extract a comparison key from each element in iterable.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.polyhedron import minlex\n    >>> minlex((1, 2, 0))\n    (0, 1, 2)\n    >>> minlex((1, 0, 2))\n    (0, 2, 1)\n    >>> minlex((1, 0, 2), directed=False)\n    (0, 1, 2)\n\n    >>> minlex('11010011000', directed=True)\n    '00011010011'\n    >>> minlex('11010011000', directed=False)\n    '00011001011'\n\n    >>> minlex(('bb', 'aaa', 'c', 'a'))\n    ('a', 'bb', 'aaa', 'c')\n    >>> minlex(('bb', 'aaa', 'c', 'a'), key=len)\n    ('c', 'a', 'bb', 'aaa')\n\n    "
    from sympy.functions.elementary.miscellaneous import Id
    if key is None:
        key = Id
    best = rotate_left(seq, least_rotation(seq, key=key))
    if not directed:
        rseq = seq[::-1]
        rbest = rotate_left(rseq, least_rotation(rseq, key=key))
        best = min(best, rbest, key=key)
    return tuple(best) if not isinstance(seq, str) else best

def runs(seq, op=gt):
    if False:
        i = 10
        return i + 15
    'Group the sequence into lists in which successive elements\n    all compare the same with the comparison operator, ``op``:\n    op(seq[i + 1], seq[i]) is True from all elements in a run.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import runs\n    >>> from operator import ge\n    >>> runs([0, 1, 2, 2, 1, 4, 3, 2, 2])\n    [[0, 1, 2], [2], [1, 4], [3], [2], [2]]\n    >>> runs([0, 1, 2, 2, 1, 4, 3, 2, 2], op=ge)\n    [[0, 1, 2, 2], [1, 4], [3], [2, 2]]\n    '
    cycles = []
    seq = iter(seq)
    try:
        run = [next(seq)]
    except StopIteration:
        return []
    while True:
        try:
            ei = next(seq)
        except StopIteration:
            break
        if op(ei, run[-1]):
            run.append(ei)
            continue
        else:
            cycles.append(run)
            run = [ei]
    if run:
        cycles.append(run)
    return cycles

def sequence_partitions(l, n, /):
    if False:
        while True:
            i = 10
    "Returns the partition of sequence $l$ into $n$ bins\n\n    Explanation\n    ===========\n\n    Given the sequence $l_1 \\cdots l_m \\in V^+$ where\n    $V^+$ is the Kleene plus of $V$\n\n    The set of $n$ partitions of $l$ is defined as:\n\n    .. math::\n        \\{(s_1, \\cdots, s_n) | s_1 \\in V^+, \\cdots, s_n \\in V^+,\n        s_1 \\cdots s_n = l_1 \\cdots l_m\\}\n\n    Parameters\n    ==========\n\n    l : Sequence[T]\n        A nonempty sequence of any Python objects\n\n    n : int\n        A positive integer\n\n    Yields\n    ======\n\n    out : list[Sequence[T]]\n        A list of sequences with concatenation equals $l$.\n        This should conform with the type of $l$.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import sequence_partitions\n    >>> for out in sequence_partitions([1, 2, 3, 4], 2):\n    ...     print(out)\n    [[1], [2, 3, 4]]\n    [[1, 2], [3, 4]]\n    [[1, 2, 3], [4]]\n\n    Notes\n    =====\n\n    This is modified version of EnricoGiampieri's partition generator\n    from https://stackoverflow.com/questions/13131491/partition-n-items-into-k-bins-in-python-lazily\n\n    See Also\n    ========\n\n    sequence_partitions_empty\n    "
    if n == 1 and l:
        yield [l]
        return
    for i in range(1, len(l)):
        for part in sequence_partitions(l[i:], n - 1):
            yield ([l[:i]] + part)

def sequence_partitions_empty(l, n, /):
    if False:
        return 10
    'Returns the partition of sequence $l$ into $n$ bins with\n    empty sequence\n\n    Explanation\n    ===========\n\n    Given the sequence $l_1 \\cdots l_m \\in V^*$ where\n    $V^*$ is the Kleene star of $V$\n\n    The set of $n$ partitions of $l$ is defined as:\n\n    .. math::\n        \\{(s_1, \\cdots, s_n) | s_1 \\in V^*, \\cdots, s_n \\in V^*,\n        s_1 \\cdots s_n = l_1 \\cdots l_m\\}\n\n    There are more combinations than :func:`sequence_partitions` because\n    empty sequence can fill everywhere, so we try to provide different\n    utility for this.\n\n    Parameters\n    ==========\n\n    l : Sequence[T]\n        A sequence of any Python objects (can be possibly empty)\n\n    n : int\n        A positive integer\n\n    Yields\n    ======\n\n    out : list[Sequence[T]]\n        A list of sequences with concatenation equals $l$.\n        This should conform with the type of $l$.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import sequence_partitions_empty\n    >>> for out in sequence_partitions_empty([1, 2, 3, 4], 2):\n    ...     print(out)\n    [[], [1, 2, 3, 4]]\n    [[1], [2, 3, 4]]\n    [[1, 2], [3, 4]]\n    [[1, 2, 3], [4]]\n    [[1, 2, 3, 4], []]\n\n    See Also\n    ========\n\n    sequence_partitions\n    '
    if n < 1:
        return
    if n == 1:
        yield [l]
        return
    for i in range(0, len(l) + 1):
        for part in sequence_partitions_empty(l[i:], n - 1):
            yield ([l[:i]] + part)

def kbins(l, k, ordered=None):
    if False:
        while True:
            i = 10
    "\n    Return sequence ``l`` partitioned into ``k`` bins.\n\n    Examples\n    ========\n\n    The default is to give the items in the same order, but grouped\n    into k partitions without any reordering:\n\n    >>> from sympy.utilities.iterables import kbins\n    >>> for p in kbins(list(range(5)), 2):\n    ...     print(p)\n    ...\n    [[0], [1, 2, 3, 4]]\n    [[0, 1], [2, 3, 4]]\n    [[0, 1, 2], [3, 4]]\n    [[0, 1, 2, 3], [4]]\n\n    The ``ordered`` flag is either None (to give the simple partition\n    of the elements) or is a 2 digit integer indicating whether the order of\n    the bins and the order of the items in the bins matters. Given::\n\n        A = [[0], [1, 2]]\n        B = [[1, 2], [0]]\n        C = [[2, 1], [0]]\n        D = [[0], [2, 1]]\n\n    the following values for ``ordered`` have the shown meanings::\n\n        00 means A == B == C == D\n        01 means A == B\n        10 means A == D\n        11 means A == A\n\n    >>> for ordered_flag in [None, 0, 1, 10, 11]:\n    ...     print('ordered = %s' % ordered_flag)\n    ...     for p in kbins(list(range(3)), 2, ordered=ordered_flag):\n    ...         print('     %s' % p)\n    ...\n    ordered = None\n         [[0], [1, 2]]\n         [[0, 1], [2]]\n    ordered = 0\n         [[0, 1], [2]]\n         [[0, 2], [1]]\n         [[0], [1, 2]]\n    ordered = 1\n         [[0], [1, 2]]\n         [[0], [2, 1]]\n         [[1], [0, 2]]\n         [[1], [2, 0]]\n         [[2], [0, 1]]\n         [[2], [1, 0]]\n    ordered = 10\n         [[0, 1], [2]]\n         [[2], [0, 1]]\n         [[0, 2], [1]]\n         [[1], [0, 2]]\n         [[0], [1, 2]]\n         [[1, 2], [0]]\n    ordered = 11\n         [[0], [1, 2]]\n         [[0, 1], [2]]\n         [[0], [2, 1]]\n         [[0, 2], [1]]\n         [[1], [0, 2]]\n         [[1, 0], [2]]\n         [[1], [2, 0]]\n         [[1, 2], [0]]\n         [[2], [0, 1]]\n         [[2, 0], [1]]\n         [[2], [1, 0]]\n         [[2, 1], [0]]\n\n    See Also\n    ========\n\n    partitions, multiset_partitions\n\n    "
    if ordered is None:
        yield from sequence_partitions(l, k)
    elif ordered == 11:
        for pl in multiset_permutations(l):
            pl = list(pl)
            yield from sequence_partitions(pl, k)
    elif ordered == 0:
        yield from multiset_partitions(l, k)
    elif ordered == 10:
        for p in multiset_partitions(l, k):
            for perm in permutations(p):
                yield list(perm)
    elif ordered == 1:
        for (kgot, p) in partitions(len(l), k, size=True):
            if kgot != k:
                continue
            for li in multiset_permutations(l):
                rv = []
                i = j = 0
                li = list(li)
                for (size, multiplicity) in sorted(p.items()):
                    for m in range(multiplicity):
                        j = i + size
                        rv.append(li[i:j])
                        i = j
                yield rv
    else:
        raise ValueError('ordered must be one of 00, 01, 10 or 11, not %s' % ordered)

def permute_signs(t):
    if False:
        i = 10
        return i + 15
    'Return iterator in which the signs of non-zero elements\n    of t are permuted.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import permute_signs\n    >>> list(permute_signs((0, 1, 2)))\n    [(0, 1, 2), (0, -1, 2), (0, 1, -2), (0, -1, -2)]\n    '
    for signs in product(*[(1, -1)] * (len(t) - t.count(0))):
        signs = list(signs)
        yield type(t)([i * signs.pop() if i else i for i in t])

def signed_permutations(t):
    if False:
        for i in range(10):
            print('nop')
    'Return iterator in which the signs of non-zero elements\n    of t and the order of the elements are permuted.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import signed_permutations\n    >>> list(signed_permutations((0, 1, 2)))\n    [(0, 1, 2), (0, -1, 2), (0, 1, -2), (0, -1, -2), (0, 2, 1),\n    (0, -2, 1), (0, 2, -1), (0, -2, -1), (1, 0, 2), (-1, 0, 2),\n    (1, 0, -2), (-1, 0, -2), (1, 2, 0), (-1, 2, 0), (1, -2, 0),\n    (-1, -2, 0), (2, 0, 1), (-2, 0, 1), (2, 0, -1), (-2, 0, -1),\n    (2, 1, 0), (-2, 1, 0), (2, -1, 0), (-2, -1, 0)]\n    '
    return (type(t)(i) for j in permutations(t) for i in permute_signs(j))

def rotations(s, dir=1):
    if False:
        i = 10
        return i + 15
    'Return a generator giving the items in s as list where\n    each subsequent list has the items rotated to the left (default)\n    or right (``dir=-1``) relative to the previous list.\n\n    Examples\n    ========\n\n    >>> from sympy import rotations\n    >>> list(rotations([1,2,3]))\n    [[1, 2, 3], [2, 3, 1], [3, 1, 2]]\n    >>> list(rotations([1,2,3], -1))\n    [[1, 2, 3], [3, 1, 2], [2, 3, 1]]\n    '
    seq = list(s)
    for i in range(len(seq)):
        yield seq
        seq = rotate_left(seq, dir)

def roundrobin(*iterables):
    if False:
        i = 10
        return i + 15
    "roundrobin recipe taken from itertools documentation:\n    https://docs.python.org/3/library/itertools.html#itertools-recipes\n\n    roundrobin('ABC', 'D', 'EF') --> A D E B F C\n\n    Recipe credited to George Sakkis\n    "
    nexts = cycle((iter(it).__next__ for it in iterables))
    pending = len(iterables)
    while pending:
        try:
            for nxt in nexts:
                yield nxt()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))

class NotIterable:
    """
    Use this as mixin when creating a class which is not supposed to
    return true when iterable() is called on its instances because
    calling list() on the instance, for example, would result in
    an infinite loop.
    """
    pass

def iterable(i, exclude=(str, dict, NotIterable)):
    if False:
        while True:
            i = 10
    '\n    Return a boolean indicating whether ``i`` is SymPy iterable.\n    True also indicates that the iterator is finite, e.g. you can\n    call list(...) on the instance.\n\n    When SymPy is working with iterables, it is almost always assuming\n    that the iterable is not a string or a mapping, so those are excluded\n    by default. If you want a pure Python definition, make exclude=None. To\n    exclude multiple items, pass them as a tuple.\n\n    You can also set the _iterable attribute to True or False on your class,\n    which will override the checks here, including the exclude test.\n\n    As a rule of thumb, some SymPy functions use this to check if they should\n    recursively map over an object. If an object is technically iterable in\n    the Python sense but does not desire this behavior (e.g., because its\n    iteration is not finite, or because iteration might induce an unwanted\n    computation), it should disable it by setting the _iterable attribute to False.\n\n    See also: is_sequence\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import iterable\n    >>> from sympy import Tuple\n    >>> things = [[1], (1,), set([1]), Tuple(1), (j for j in [1, 2]), {1:2}, \'1\', 1]\n    >>> for i in things:\n    ...     print(\'%s %s\' % (iterable(i), type(i)))\n    True <... \'list\'>\n    True <... \'tuple\'>\n    True <... \'set\'>\n    True <class \'sympy.core.containers.Tuple\'>\n    True <... \'generator\'>\n    False <... \'dict\'>\n    False <... \'str\'>\n    False <... \'int\'>\n\n    >>> iterable({}, exclude=None)\n    True\n    >>> iterable({}, exclude=str)\n    True\n    >>> iterable("no", exclude=str)\n    False\n\n    '
    if hasattr(i, '_iterable'):
        return i._iterable
    try:
        iter(i)
    except TypeError:
        return False
    if exclude:
        return not isinstance(i, exclude)
    return True

def is_sequence(i, include=None):
    if False:
        i = 10
        return i + 15
    "\n    Return a boolean indicating whether ``i`` is a sequence in the SymPy\n    sense. If anything that fails the test below should be included as\n    being a sequence for your application, set 'include' to that object's\n    type; multiple types should be passed as a tuple of types.\n\n    Note: although generators can generate a sequence, they often need special\n    handling to make sure their elements are captured before the generator is\n    exhausted, so these are not included by default in the definition of a\n    sequence.\n\n    See also: iterable\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.iterables import is_sequence\n    >>> from types import GeneratorType\n    >>> is_sequence([])\n    True\n    >>> is_sequence(set())\n    False\n    >>> is_sequence('abc')\n    False\n    >>> is_sequence('abc', include=str)\n    True\n    >>> generator = (c for c in 'abc')\n    >>> is_sequence(generator)\n    False\n    >>> is_sequence(generator, include=(str, GeneratorType))\n    True\n\n    "
    return hasattr(i, '__getitem__') and iterable(i) or (bool(include) and isinstance(i, include))

@deprecated('\n    Using postorder_traversal from the sympy.utilities.iterables submodule is\n    deprecated.\n\n    Instead, use postorder_traversal from the top-level sympy namespace, like\n\n        sympy.postorder_traversal\n    ', deprecated_since_version='1.10', active_deprecations_target='deprecated-traversal-functions-moved')
def postorder_traversal(node, keys=None):
    if False:
        while True:
            i = 10
    from sympy.core.traversal import postorder_traversal as _postorder_traversal
    return _postorder_traversal(node, keys=keys)

@deprecated('\n    Using interactive_traversal from the sympy.utilities.iterables submodule\n    is deprecated.\n\n    Instead, use interactive_traversal from the top-level sympy namespace,\n    like\n\n        sympy.interactive_traversal\n    ', deprecated_since_version='1.10', active_deprecations_target='deprecated-traversal-functions-moved')
def interactive_traversal(expr):
    if False:
        return 10
    from sympy.interactive.traversal import interactive_traversal as _interactive_traversal
    return _interactive_traversal(expr)

@deprecated('\n    Importing default_sort_key from sympy.utilities.iterables is deprecated.\n    Use from sympy import default_sort_key instead.\n    ', deprecated_since_version='1.10', active_deprecations_target='deprecated-sympy-core-compatibility')
def default_sort_key(*args, **kwargs):
    if False:
        print('Hello World!')
    from sympy import default_sort_key as _default_sort_key
    return _default_sort_key(*args, **kwargs)

@deprecated('\n    Importing default_sort_key from sympy.utilities.iterables is deprecated.\n    Use from sympy import default_sort_key instead.\n    ', deprecated_since_version='1.10', active_deprecations_target='deprecated-sympy-core-compatibility')
def ordered(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    from sympy import ordered as _ordered
    return _ordered(*args, **kwargs)