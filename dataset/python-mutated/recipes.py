"""Imported from the recipes section of the itertools documentation.

All functions taken from the recipes section of the itertools library docs
[1]_.
Some backward-compatible usability improvements have been made.

.. [1] http://docs.python.org/library/itertools.html#recipes

"""
import math
import operator
from collections import deque
from collections.abc import Sized
from functools import partial, reduce
from itertools import chain, combinations, compress, count, cycle, groupby, islice, product, repeat, starmap, tee, zip_longest
from random import randrange, sample, choice
__all__ = ['all_equal', 'batched', 'before_and_after', 'consume', 'convolve', 'dotproduct', 'first_true', 'factor', 'flatten', 'grouper', 'iter_except', 'iter_index', 'matmul', 'ncycles', 'nth', 'nth_combination', 'padnone', 'pad_none', 'pairwise', 'partition', 'polynomial_eval', 'polynomial_from_roots', 'polynomial_derivative', 'powerset', 'prepend', 'quantify', 'random_combination_with_replacement', 'random_combination', 'random_permutation', 'random_product', 'repeatfunc', 'roundrobin', 'sieve', 'sliding_window', 'subslices', 'sum_of_squares', 'tabulate', 'tail', 'take', 'transpose', 'triplewise', 'unique_everseen', 'unique_justseen']
_marker = object()
try:
    zip(strict=True)
except TypeError:
    _zip_strict = zip
else:
    _zip_strict = partial(zip, strict=True)
_sumprod = getattr(math, 'sumprod', lambda x, y: dotproduct(x, y))

def take(n, iterable):
    if False:
        for i in range(10):
            print('nop')
    'Return first *n* items of the iterable as a list.\n\n        >>> take(3, range(10))\n        [0, 1, 2]\n\n    If there are fewer than *n* items in the iterable, all of them are\n    returned.\n\n        >>> take(10, range(3))\n        [0, 1, 2]\n\n    '
    return list(islice(iterable, n))

def tabulate(function, start=0):
    if False:
        i = 10
        return i + 15
    'Return an iterator over the results of ``func(start)``,\n    ``func(start + 1)``, ``func(start + 2)``...\n\n    *func* should be a function that accepts one integer argument.\n\n    If *start* is not specified it defaults to 0. It will be incremented each\n    time the iterator is advanced.\n\n        >>> square = lambda x: x ** 2\n        >>> iterator = tabulate(square, -3)\n        >>> take(4, iterator)\n        [9, 4, 1, 0]\n\n    '
    return map(function, count(start))

def tail(n, iterable):
    if False:
        return 10
    "Return an iterator over the last *n* items of *iterable*.\n\n    >>> t = tail(3, 'ABCDEFG')\n    >>> list(t)\n    ['E', 'F', 'G']\n\n    "
    if isinstance(iterable, Sized):
        yield from islice(iterable, max(0, len(iterable) - n), None)
    else:
        yield from iter(deque(iterable, maxlen=n))

def consume(iterator, n=None):
    if False:
        for i in range(10):
            print('nop')
    'Advance *iterable* by *n* steps. If *n* is ``None``, consume it\n    entirely.\n\n    Efficiently exhausts an iterator without returning values. Defaults to\n    consuming the whole iterator, but an optional second argument may be\n    provided to limit consumption.\n\n        >>> i = (x for x in range(10))\n        >>> next(i)\n        0\n        >>> consume(i, 3)\n        >>> next(i)\n        4\n        >>> consume(i)\n        >>> next(i)\n        Traceback (most recent call last):\n          File "<stdin>", line 1, in <module>\n        StopIteration\n\n    If the iterator has fewer items remaining than the provided limit, the\n    whole iterator will be consumed.\n\n        >>> i = (x for x in range(3))\n        >>> consume(i, 5)\n        >>> next(i)\n        Traceback (most recent call last):\n          File "<stdin>", line 1, in <module>\n        StopIteration\n\n    '
    if n is None:
        deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)

def nth(iterable, n, default=None):
    if False:
        return 10
    'Returns the nth item or a default value.\n\n    >>> l = range(10)\n    >>> nth(l, 3)\n    3\n    >>> nth(l, 20, "zebra")\n    \'zebra\'\n\n    '
    return next(islice(iterable, n, None), default)

def all_equal(iterable):
    if False:
        while True:
            i = 10
    "\n    Returns ``True`` if all the elements are equal to each other.\n\n        >>> all_equal('aaaa')\n        True\n        >>> all_equal('aaab')\n        False\n\n    "
    g = groupby(iterable)
    return next(g, True) and (not next(g, False))

def quantify(iterable, pred=bool):
    if False:
        print('Hello World!')
    'Return the how many times the predicate is true.\n\n    >>> quantify([True, False, True])\n    2\n\n    '
    return sum(map(pred, iterable))

def pad_none(iterable):
    if False:
        return 10
    'Returns the sequence of elements and then returns ``None`` indefinitely.\n\n        >>> take(5, pad_none(range(3)))\n        [0, 1, 2, None, None]\n\n    Useful for emulating the behavior of the built-in :func:`map` function.\n\n    See also :func:`padded`.\n\n    '
    return chain(iterable, repeat(None))
padnone = pad_none

def ncycles(iterable, n):
    if False:
        for i in range(10):
            print('nop')
    'Returns the sequence elements *n* times\n\n    >>> list(ncycles(["a", "b"], 3))\n    [\'a\', \'b\', \'a\', \'b\', \'a\', \'b\']\n\n    '
    return chain.from_iterable(repeat(tuple(iterable), n))

def dotproduct(vec1, vec2):
    if False:
        for i in range(10):
            print('nop')
    'Returns the dot product of the two iterables.\n\n    >>> dotproduct([10, 10], [20, 20])\n    400\n\n    '
    return sum(map(operator.mul, vec1, vec2))

def flatten(listOfLists):
    if False:
        print('Hello World!')
    'Return an iterator flattening one level of nesting in a list of lists.\n\n        >>> list(flatten([[0, 1], [2, 3]]))\n        [0, 1, 2, 3]\n\n    See also :func:`collapse`, which can flatten multiple levels of nesting.\n\n    '
    return chain.from_iterable(listOfLists)

def repeatfunc(func, times=None, *args):
    if False:
        return 10
    'Call *func* with *args* repeatedly, returning an iterable over the\n    results.\n\n    If *times* is specified, the iterable will terminate after that many\n    repetitions:\n\n        >>> from operator import add\n        >>> times = 4\n        >>> args = 3, 5\n        >>> list(repeatfunc(add, times, *args))\n        [8, 8, 8, 8]\n\n    If *times* is ``None`` the iterable will not terminate:\n\n        >>> from random import randrange\n        >>> times = None\n        >>> args = 1, 11\n        >>> take(6, repeatfunc(randrange, times, *args))  # doctest:+SKIP\n        [2, 4, 8, 1, 8, 4]\n\n    '
    if times is None:
        return starmap(func, repeat(args))
    return starmap(func, repeat(args, times))

def _pairwise(iterable):
    if False:
        return 10
    'Returns an iterator of paired items, overlapping, from the original\n\n    >>> take(4, pairwise(count()))\n    [(0, 1), (1, 2), (2, 3), (3, 4)]\n\n    On Python 3.10 and above, this is an alias for :func:`itertools.pairwise`.\n\n    '
    (a, b) = tee(iterable)
    next(b, None)
    return zip(a, b)
try:
    from itertools import pairwise as itertools_pairwise
except ImportError:
    pairwise = _pairwise
else:

    def pairwise(iterable):
        if False:
            for i in range(10):
                print('nop')
        return itertools_pairwise(iterable)
    pairwise.__doc__ = _pairwise.__doc__

class UnequalIterablesError(ValueError):

    def __init__(self, details=None):
        if False:
            for i in range(10):
                print('nop')
        msg = 'Iterables have different lengths'
        if details is not None:
            msg += ': index 0 has length {}; index {} has length {}'.format(*details)
        super().__init__(msg)

def _zip_equal_generator(iterables):
    if False:
        i = 10
        return i + 15
    for combo in zip_longest(*iterables, fillvalue=_marker):
        for val in combo:
            if val is _marker:
                raise UnequalIterablesError()
        yield combo

def _zip_equal(*iterables):
    if False:
        print('Hello World!')
    try:
        first_size = len(iterables[0])
        for (i, it) in enumerate(iterables[1:], 1):
            size = len(it)
            if size != first_size:
                raise UnequalIterablesError(details=(first_size, i, size))
        return zip(*iterables)
    except TypeError:
        return _zip_equal_generator(iterables)

def grouper(iterable, n, incomplete='fill', fillvalue=None):
    if False:
        i = 10
        return i + 15
    "Group elements from *iterable* into fixed-length groups of length *n*.\n\n    >>> list(grouper('ABCDEF', 3))\n    [('A', 'B', 'C'), ('D', 'E', 'F')]\n\n    The keyword arguments *incomplete* and *fillvalue* control what happens for\n    iterables whose length is not a multiple of *n*.\n\n    When *incomplete* is `'fill'`, the last group will contain instances of\n    *fillvalue*.\n\n    >>> list(grouper('ABCDEFG', 3, incomplete='fill', fillvalue='x'))\n    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G', 'x', 'x')]\n\n    When *incomplete* is `'ignore'`, the last group will not be emitted.\n\n    >>> list(grouper('ABCDEFG', 3, incomplete='ignore', fillvalue='x'))\n    [('A', 'B', 'C'), ('D', 'E', 'F')]\n\n    When *incomplete* is `'strict'`, a subclass of `ValueError` will be raised.\n\n    >>> it = grouper('ABCDEFG', 3, incomplete='strict')\n    >>> list(it)  # doctest: +IGNORE_EXCEPTION_DETAIL\n    Traceback (most recent call last):\n    ...\n    UnequalIterablesError\n\n    "
    args = [iter(iterable)] * n
    if incomplete == 'fill':
        return zip_longest(*args, fillvalue=fillvalue)
    if incomplete == 'strict':
        return _zip_equal(*args)
    if incomplete == 'ignore':
        return zip(*args)
    else:
        raise ValueError('Expected fill, strict, or ignore')

def roundrobin(*iterables):
    if False:
        print('Hello World!')
    "Yields an item from each iterable, alternating between them.\n\n        >>> list(roundrobin('ABC', 'D', 'EF'))\n        ['A', 'D', 'E', 'B', 'F', 'C']\n\n    This function produces the same output as :func:`interleave_longest`, but\n    may perform better for some inputs (in particular when the number of\n    iterables is small).\n\n    "
    pending = len(iterables)
    nexts = cycle((iter(it).__next__ for it in iterables))
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))

def partition(pred, iterable):
    if False:
        return 10
    "\n    Returns a 2-tuple of iterables derived from the input iterable.\n    The first yields the items that have ``pred(item) == False``.\n    The second yields the items that have ``pred(item) == True``.\n\n        >>> is_odd = lambda x: x % 2 != 0\n        >>> iterable = range(10)\n        >>> even_items, odd_items = partition(is_odd, iterable)\n        >>> list(even_items), list(odd_items)\n        ([0, 2, 4, 6, 8], [1, 3, 5, 7, 9])\n\n    If *pred* is None, :func:`bool` is used.\n\n        >>> iterable = [0, 1, False, True, '', ' ']\n        >>> false_items, true_items = partition(None, iterable)\n        >>> list(false_items), list(true_items)\n        ([0, False, ''], [1, True, ' '])\n\n    "
    if pred is None:
        pred = bool
    (t1, t2, p) = tee(iterable, 3)
    (p1, p2) = tee(map(pred, p))
    return (compress(t1, map(operator.not_, p1)), compress(t2, p2))

def powerset(iterable):
    if False:
        print('Hello World!')
    "Yields all possible subsets of the iterable.\n\n        >>> list(powerset([1, 2, 3]))\n        [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]\n\n    :func:`powerset` will operate on iterables that aren't :class:`set`\n    instances, so repeated elements in the input will produce repeated elements\n    in the output. Use :func:`unique_everseen` on the input to avoid generating\n    duplicates:\n\n        >>> seq = [1, 1, 0]\n        >>> list(powerset(seq))\n        [(), (1,), (1,), (0,), (1, 1), (1, 0), (1, 0), (1, 1, 0)]\n        >>> from more_itertools import unique_everseen\n        >>> list(powerset(unique_everseen(seq)))\n        [(), (1,), (0,), (1, 0)]\n\n    "
    s = list(iterable)
    return chain.from_iterable((combinations(s, r) for r in range(len(s) + 1)))

def unique_everseen(iterable, key=None):
    if False:
        while True:
            i = 10
    "\n    Yield unique elements, preserving order.\n\n        >>> list(unique_everseen('AAAABBBCCDAABBB'))\n        ['A', 'B', 'C', 'D']\n        >>> list(unique_everseen('ABBCcAD', str.lower))\n        ['A', 'B', 'C', 'D']\n\n    Sequences with a mix of hashable and unhashable items can be used.\n    The function will be slower (i.e., `O(n^2)`) for unhashable items.\n\n    Remember that ``list`` objects are unhashable - you can use the *key*\n    parameter to transform the list to a tuple (which is hashable) to\n    avoid a slowdown.\n\n        >>> iterable = ([1, 2], [2, 3], [1, 2])\n        >>> list(unique_everseen(iterable))  # Slow\n        [[1, 2], [2, 3]]\n        >>> list(unique_everseen(iterable, key=tuple))  # Faster\n        [[1, 2], [2, 3]]\n\n    Similary, you may want to convert unhashable ``set`` objects with\n    ``key=frozenset``. For ``dict`` objects,\n    ``key=lambda x: frozenset(x.items())`` can be used.\n\n    "
    seenset = set()
    seenset_add = seenset.add
    seenlist = []
    seenlist_add = seenlist.append
    use_key = key is not None
    for element in iterable:
        k = key(element) if use_key else element
        try:
            if k not in seenset:
                seenset_add(k)
                yield element
        except TypeError:
            if k not in seenlist:
                seenlist_add(k)
                yield element

def unique_justseen(iterable, key=None):
    if False:
        for i in range(10):
            print('nop')
    "Yields elements in order, ignoring serial duplicates\n\n    >>> list(unique_justseen('AAAABBBCCDAABBB'))\n    ['A', 'B', 'C', 'D', 'A', 'B']\n    >>> list(unique_justseen('ABBCcAD', str.lower))\n    ['A', 'B', 'C', 'A', 'D']\n\n    "
    return map(next, map(operator.itemgetter(1), groupby(iterable, key)))

def iter_except(func, exception, first=None):
    if False:
        while True:
            i = 10
    "Yields results from a function repeatedly until an exception is raised.\n\n    Converts a call-until-exception interface to an iterator interface.\n    Like ``iter(func, sentinel)``, but uses an exception instead of a sentinel\n    to end the loop.\n\n        >>> l = [0, 1, 2]\n        >>> list(iter_except(l.pop, IndexError))\n        [2, 1, 0]\n\n    Multiple exceptions can be specified as a stopping condition:\n\n        >>> l = [1, 2, 3, '...', 4, 5, 6]\n        >>> list(iter_except(lambda: 1 + l.pop(), (IndexError, TypeError)))\n        [7, 6, 5]\n        >>> list(iter_except(lambda: 1 + l.pop(), (IndexError, TypeError)))\n        [4, 3, 2]\n        >>> list(iter_except(lambda: 1 + l.pop(), (IndexError, TypeError)))\n        []\n\n    "
    try:
        if first is not None:
            yield first()
        while 1:
            yield func()
    except exception:
        pass

def first_true(iterable, default=None, pred=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns the first true value in the iterable.\n\n    If no true value is found, returns *default*\n\n    If *pred* is not None, returns the first item for which\n    ``pred(item) == True`` .\n\n        >>> first_true(range(10))\n        1\n        >>> first_true(range(10), pred=lambda x: x > 5)\n        6\n        >>> first_true(range(10), default='missing', pred=lambda x: x > 9)\n        'missing'\n\n    "
    return next(filter(pred, iterable), default)

def random_product(*args, repeat=1):
    if False:
        return 10
    "Draw an item at random from each of the input iterables.\n\n        >>> random_product('abc', range(4), 'XYZ')  # doctest:+SKIP\n        ('c', 3, 'Z')\n\n    If *repeat* is provided as a keyword argument, that many items will be\n    drawn from each iterable.\n\n        >>> random_product('abcd', range(4), repeat=2)  # doctest:+SKIP\n        ('a', 2, 'd', 3)\n\n    This equivalent to taking a random selection from\n    ``itertools.product(*args, **kwarg)``.\n\n    "
    pools = [tuple(pool) for pool in args] * repeat
    return tuple((choice(pool) for pool in pools))

def random_permutation(iterable, r=None):
    if False:
        return 10
    'Return a random *r* length permutation of the elements in *iterable*.\n\n    If *r* is not specified or is ``None``, then *r* defaults to the length of\n    *iterable*.\n\n        >>> random_permutation(range(5))  # doctest:+SKIP\n        (3, 4, 0, 1, 2)\n\n    This equivalent to taking a random selection from\n    ``itertools.permutations(iterable, r)``.\n\n    '
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(sample(pool, r))

def random_combination(iterable, r):
    if False:
        for i in range(10):
            print('nop')
    'Return a random *r* length subsequence of the elements in *iterable*.\n\n        >>> random_combination(range(5), 3)  # doctest:+SKIP\n        (2, 3, 4)\n\n    This equivalent to taking a random selection from\n    ``itertools.combinations(iterable, r)``.\n\n    '
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(sample(range(n), r))
    return tuple((pool[i] for i in indices))

def random_combination_with_replacement(iterable, r):
    if False:
        i = 10
        return i + 15
    'Return a random *r* length subsequence of elements in *iterable*,\n    allowing individual elements to be repeated.\n\n        >>> random_combination_with_replacement(range(3), 5) # doctest:+SKIP\n        (0, 0, 1, 2, 2)\n\n    This equivalent to taking a random selection from\n    ``itertools.combinations_with_replacement(iterable, r)``.\n\n    '
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted((randrange(n) for i in range(r)))
    return tuple((pool[i] for i in indices))

def nth_combination(iterable, r, index):
    if False:
        return 10
    'Equivalent to ``list(combinations(iterable, r))[index]``.\n\n    The subsequences of *iterable* that are of length *r* can be ordered\n    lexicographically. :func:`nth_combination` computes the subsequence at\n    sort position *index* directly, without computing the previous\n    subsequences.\n\n        >>> nth_combination(range(5), 3, 5)\n        (0, 3, 4)\n\n    ``ValueError`` will be raised If *r* is negative or greater than the length\n    of *iterable*.\n    ``IndexError`` will be raised if the given *index* is invalid.\n    '
    pool = tuple(iterable)
    n = len(pool)
    if r < 0 or r > n:
        raise ValueError
    c = 1
    k = min(r, n - r)
    for i in range(1, k + 1):
        c = c * (n - k + i) // i
    if index < 0:
        index += c
    if index < 0 or index >= c:
        raise IndexError
    result = []
    while r:
        (c, n, r) = (c * r // n, n - 1, r - 1)
        while index >= c:
            index -= c
            (c, n) = (c * (n - r) // n, n - 1)
        result.append(pool[-1 - n])
    return tuple(result)

def prepend(value, iterator):
    if False:
        print('Hello World!')
    "Yield *value*, followed by the elements in *iterator*.\n\n        >>> value = '0'\n        >>> iterator = ['1', '2', '3']\n        >>> list(prepend(value, iterator))\n        ['0', '1', '2', '3']\n\n    To prepend multiple values, see :func:`itertools.chain`\n    or :func:`value_chain`.\n\n    "
    return chain([value], iterator)

def convolve(signal, kernel):
    if False:
        return 10
    'Convolve the iterable *signal* with the iterable *kernel*.\n\n        >>> signal = (1, 2, 3, 4, 5)\n        >>> kernel = [3, 2, 1]\n        >>> list(convolve(signal, kernel))\n        [3, 8, 14, 20, 26, 14, 5]\n\n    Note: the input arguments are not interchangeable, as the *kernel*\n    is immediately consumed and stored.\n\n    '
    kernel = tuple(kernel)[::-1]
    n = len(kernel)
    window = deque([0], maxlen=n) * n
    for x in chain(signal, repeat(0, n - 1)):
        window.append(x)
        yield _sumprod(kernel, window)

def before_and_after(predicate, it):
    if False:
        i = 10
        return i + 15
    "A variant of :func:`takewhile` that allows complete access to the\n    remainder of the iterator.\n\n         >>> it = iter('ABCdEfGhI')\n         >>> all_upper, remainder = before_and_after(str.isupper, it)\n         >>> ''.join(all_upper)\n         'ABC'\n         >>> ''.join(remainder) # takewhile() would lose the 'd'\n         'dEfGhI'\n\n    Note that the first iterator must be fully consumed before the second\n    iterator can generate valid results.\n    "
    it = iter(it)
    transition = []

    def true_iterator():
        if False:
            for i in range(10):
                print('nop')
        for elem in it:
            if predicate(elem):
                yield elem
            else:
                transition.append(elem)
                return
    remainder_iterator = chain(transition, it)
    return (true_iterator(), remainder_iterator)

def triplewise(iterable):
    if False:
        while True:
            i = 10
    "Return overlapping triplets from *iterable*.\n\n    >>> list(triplewise('ABCDE'))\n    [('A', 'B', 'C'), ('B', 'C', 'D'), ('C', 'D', 'E')]\n\n    "
    for ((a, _), (b, c)) in pairwise(pairwise(iterable)):
        yield (a, b, c)

def sliding_window(iterable, n):
    if False:
        return 10
    'Return a sliding window of width *n* over *iterable*.\n\n        >>> list(sliding_window(range(6), 4))\n        [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)]\n\n    If *iterable* has fewer than *n* items, then nothing is yielded:\n\n        >>> list(sliding_window(range(3), 4))\n        []\n\n    For a variant with more features, see :func:`windowed`.\n    '
    it = iter(iterable)
    window = deque(islice(it, n - 1), maxlen=n)
    for x in it:
        window.append(x)
        yield tuple(window)

def subslices(iterable):
    if False:
        for i in range(10):
            print('nop')
    "Return all contiguous non-empty subslices of *iterable*.\n\n        >>> list(subslices('ABC'))\n        [['A'], ['A', 'B'], ['A', 'B', 'C'], ['B'], ['B', 'C'], ['C']]\n\n    This is similar to :func:`substrings`, but emits items in a different\n    order.\n    "
    seq = list(iterable)
    slices = starmap(slice, combinations(range(len(seq) + 1), 2))
    return map(operator.getitem, repeat(seq), slices)

def polynomial_from_roots(roots):
    if False:
        i = 10
        return i + 15
    "Compute a polynomial's coefficients from its roots.\n\n    >>> roots = [5, -4, 3]  # (x - 5) * (x + 4) * (x - 3)\n    >>> polynomial_from_roots(roots)  # x^3 - 4 * x^2 - 17 * x + 60\n    [1, -4, -17, 60]\n    "
    factors = zip(repeat(1), map(operator.neg, roots))
    return list(reduce(convolve, factors, [1]))

def iter_index(iterable, value, start=0):
    if False:
        for i in range(10):
            print('nop')
    "Yield the index of each place in *iterable* that *value* occurs,\n    beginning with index *start*.\n\n    See :func:`locate` for a more general means of finding the indexes\n    associated with particular values.\n\n    >>> list(iter_index('AABCADEAF', 'A'))\n    [0, 1, 4, 7]\n    "
    try:
        seq_index = iterable.index
    except AttributeError:
        it = islice(iterable, start, None)
        i = start - 1
        try:
            while True:
                i = i + operator.indexOf(it, value) + 1
                yield i
        except ValueError:
            pass
    else:
        i = start - 1
        try:
            while True:
                i = seq_index(value, i + 1)
                yield i
        except ValueError:
            pass

def sieve(n):
    if False:
        i = 10
        return i + 15
    'Yield the primes less than n.\n\n    >>> list(sieve(30))\n    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]\n    '
    data = bytearray((0, 1)) * (n // 2)
    data[:3] = (0, 0, 0)
    limit = math.isqrt(n) + 1
    for p in compress(range(limit), data):
        data[p * p:n:p + p] = bytes(len(range(p * p, n, p + p)))
    data[2] = 1
    return iter_index(data, 1) if n > 2 else iter([])

def _batched(iterable, n):
    if False:
        i = 10
        return i + 15
    "Batch data into lists of length *n*. The last batch may be shorter.\n\n    >>> list(batched('ABCDEFG', 3))\n    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]\n\n    On Python 3.12 and above, this is an alias for :func:`itertools.batched`.\n    "
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if not batch:
            break
        yield batch
try:
    from itertools import batched as itertools_batched
except ImportError:
    batched = _batched
else:

    def batched(iterable, n):
        if False:
            i = 10
            return i + 15
        return itertools_batched(iterable, n)
    batched.__doc__ = _batched.__doc__

def transpose(it):
    if False:
        while True:
            i = 10
    'Swap the rows and columns of the input.\n\n    >>> list(transpose([(1, 2, 3), (11, 22, 33)]))\n    [(1, 11), (2, 22), (3, 33)]\n\n    The caller should ensure that the dimensions of the input are compatible.\n    If the input is empty, no output will be produced.\n    '
    return _zip_strict(*it)

def matmul(m1, m2):
    if False:
        while True:
            i = 10
    'Multiply two matrices.\n    >>> list(matmul([(7, 5), (3, 5)], [(2, 5), (7, 9)]))\n    [(49, 80), (41, 60)]\n\n    The caller should ensure that the dimensions of the input matrices are\n    compatible with each other.\n    '
    n = len(m2[0])
    return batched(starmap(_sumprod, product(m1, transpose(m2))), n)

def factor(n):
    if False:
        while True:
            i = 10
    'Yield the prime factors of n.\n    >>> list(factor(360))\n    [2, 2, 2, 3, 3, 5]\n    '
    for prime in sieve(math.isqrt(n) + 1):
        while True:
            if n % prime:
                break
            yield prime
            n //= prime
            if n == 1:
                return
    if n > 1:
        yield n

def polynomial_eval(coefficients, x):
    if False:
        for i in range(10):
            print('nop')
    'Evaluate a polynomial at a specific value.\n\n    Example: evaluating x^3 - 4 * x^2 - 17 * x + 60 at x = 2.5:\n\n    >>> coefficients = [1, -4, -17, 60]\n    >>> x = 2.5\n    >>> polynomial_eval(coefficients, x)\n    8.125\n    '
    n = len(coefficients)
    if n == 0:
        return x * 0
    powers = map(pow, repeat(x), reversed(range(n)))
    return _sumprod(coefficients, powers)

def sum_of_squares(it):
    if False:
        while True:
            i = 10
    'Return the sum of the squares of the input values.\n\n    >>> sum_of_squares([10, 20, 30])\n    1400\n    '
    return _sumprod(*tee(it))

def polynomial_derivative(coefficients):
    if False:
        for i in range(10):
            print('nop')
    'Compute the first derivative of a polynomial.\n\n    Example: evaluating the derivative of x^3 - 4 * x^2 - 17 * x + 60\n\n    >>> coefficients = [1, -4, -17, 60]\n    >>> derivative_coefficients = polynomial_derivative(coefficients)\n    >>> derivative_coefficients\n    [3, -8, -17]\n    '
    n = len(coefficients)
    powers = reversed(range(1, n))
    return list(map(operator.mul, coefficients, powers))