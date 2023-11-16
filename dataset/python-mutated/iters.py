"""
This module includes and extends the standard module :mod:`itertools`.
"""
from __future__ import absolute_import
from __future__ import division
import collections
import copy
import multiprocessing
import operator
import random
import time
from itertools import *
from six.moves import map, filter, filterfalse, range, zip, zip_longest
from pwnlib.context import context
from pwnlib.log import getLogger
__all__ = ['bruteforce', 'mbruteforce', 'chained', 'consume', 'cyclen', 'dotproduct', 'flatten', 'group', 'iter_except', 'lexicographic', 'lookahead', 'nth', 'pad', 'pairwise', 'powerset', 'quantify', 'random_combination', 'random_combination_with_replacement', 'random_permutation', 'random_product', 'repeat_func', 'roundrobin', 'tabulate', 'take', 'unique_everseen', 'unique_justseen', 'unique_window', 'chain', 'combinations', 'combinations_with_replacement', 'compress', 'count', 'cycle', 'dropwhile', 'groupby', 'filter', 'filterfalse', 'map', 'islice', 'zip', 'zip_longest', 'permutations', 'product', 'repeat', 'starmap', 'takewhile', 'tee']
log = getLogger(__name__)

def take(n, iterable):
    if False:
        for i in range(10):
            print('nop')
    'take(n, iterable) -> list\n\n    Returns first `n` elements of `iterable`.  If `iterable` is a iterator it\n    will be advanced.\n\n    Arguments:\n      n(int):  Number of elements to take.\n      iterable:  An iterable.\n\n    Returns:\n      A list of the first `n` elements of `iterable`.  If there are fewer than\n      `n` elements in `iterable` they will all be returned.\n\n    Examples:\n      >>> take(2, range(10))\n      [0, 1]\n      >>> i = count()\n      >>> take(2, i)\n      [0, 1]\n      >>> take(2, i)\n      [2, 3]\n      >>> take(9001, [1, 2, 3])\n      [1, 2, 3]\n    '
    return list(islice(iterable, n))

def tabulate(func, start=0):
    if False:
        for i in range(10):
            print('nop')
    "tabulate(func, start = 0) -> iterator\n\n    Arguments:\n      func(function):  The function to tabulate over.\n      start(int):  Number to start on.\n\n    Returns:\n      An iterator with the elements ``func(start), func(start + 1), ...``.\n\n    Examples:\n      >>> take(2, tabulate(str))\n      ['0', '1']\n      >>> take(5, tabulate(lambda x: x**2, start = 1))\n      [1, 4, 9, 16, 25]\n    "
    return map(func, count(start))

def consume(n, iterator):
    if False:
        for i in range(10):
            print('nop')
    'consume(n, iterator)\n\n    Advance the iterator `n` steps ahead. If `n is :const:`None`, consume\n    everything.\n\n    Arguments:\n      n(int):  Number of elements to consume.\n      iterator(iterator):  An iterator.\n\n    Returns:\n      :const:`None`.\n\n    Examples:\n      >>> i = count()\n      >>> consume(5, i)\n      >>> next(i)\n      5\n      >>> i = iter([1, 2, 3, 4, 5])\n      >>> consume(2, i)\n      >>> list(i)\n      [3, 4, 5]\n      >>> def g():\n      ...     for i in range(2):\n      ...         yield i\n      ...         print(i)\n      >>> consume(None, g())\n      0\n      1\n    '
    if n is None:
        collections.deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)

def nth(n, iterable, default=None):
    if False:
        while True:
            i = 10
    'nth(n, iterable, default = None) -> object\n\n    Returns the element at index `n` in `iterable`.  If `iterable` is a\n    iterator it will be advanced.\n\n    Arguments:\n      n(int):  Index of the element to return.\n      iterable:  An iterable.\n      default(objext):  A default value.\n\n    Returns:\n      The element at index `n` in `iterable` or `default` if `iterable` has too\n      few elements.\n\n    Examples:\n      >>> nth(2, [0, 1, 2, 3])\n      2\n      >>> nth(2, [0, 1], 42)\n      42\n      >>> i = count()\n      >>> nth(42, i)\n      42\n      >>> nth(42, i)\n      85\n    '
    return next(islice(iterable, n, None), default)

def quantify(iterable, pred=bool):
    if False:
        print('Hello World!')
    "quantify(iterable, pred = bool) -> int\n\n    Count how many times the predicate `pred` is :const:`True`.\n\n    Arguments:\n        iterable:  An iterable.\n        pred:  A function that given an element from `iterable` returns either\n               :const:`True` or :const:`False`.\n\n    Returns:\n      The number of elements in `iterable` for which `pred` returns\n      :const:`True`.\n\n    Examples:\n      >>> quantify([1, 2, 3, 4], lambda x: x % 2 == 0)\n      2\n      >>> quantify(['1', 'two', '3', '42'], str.isdigit)\n      3\n    "
    return sum(map(pred, iterable))

def pad(iterable, value=None):
    if False:
        return 10
    'pad(iterable, value = None) -> iterator\n\n    Pad an `iterable` with `value`, i.e. returns an iterator whoose elements are\n    first the elements of `iterable` then `value` indefinitely.\n\n    Arguments:\n      iterable:  An iterable.\n      value:  The value to pad with.\n\n    Returns:\n      An iterator whoose elements are first the elements of `iterable` then\n      `value` indefinitely.\n\n    Examples:\n      >>> take(3, pad([1, 2]))\n      [1, 2, None]\n      >>> i = pad(iter([1, 2, 3]), 42)\n      >>> take(2, i)\n      [1, 2]\n      >>> take(2, i)\n      [3, 42]\n      >>> take(2, i)\n      [42, 42]\n    '
    return chain(iterable, repeat(value))

def cyclen(n, iterable):
    if False:
        return 10
    'cyclen(n, iterable) -> iterator\n\n    Repeats the elements of `iterable` `n` times.\n\n    Arguments:\n      n(int):  The number of times to repeat `iterable`.\n      iterable:  An iterable.\n\n    Returns:\n      An iterator whoose elements are the elements of `iterator` repeated `n`\n      times.\n\n    Examples:\n      >>> take(4, cyclen(2, [1, 2]))\n      [1, 2, 1, 2]\n      >>> list(cyclen(10, []))\n      []\n    '
    return chain.from_iterable(repeat(tuple(iterable), n))

def dotproduct(x, y):
    if False:
        while True:
            i = 10
    'dotproduct(x, y) -> int\n\n    Computes the dot product of `x` and `y`.\n\n    Arguments:\n      x(iterable):  An iterable.\n      x(iterable):  An iterable.\n\n    Returns:\n      The dot product of `x` and `y`, i.e.: ``x[0] * y[0] + x[1] * y[1] + ...``.\n\n    Example:\n      >>> dotproduct([1, 2, 3], [4, 5, 6])\n      ... # 1 * 4 + 2 * 5 + 3 * 6 == 32\n      32\n    '
    return sum(map(operator.mul, x, y))

def flatten(xss):
    if False:
        i = 10
        return i + 15
    'flatten(xss) -> iterator\n\n    Flattens one level of nesting; when `xss` is an iterable of iterables,\n    returns an iterator whoose elements is the concatenation of the elements of\n    `xss`.\n\n    Arguments:\n      xss:  An iterable of iterables.\n\n    Returns:\n      An iterator whoose elements are the concatenation of the iterables in\n      `xss`.\n\n    Examples:\n      >>> list(flatten([[1, 2], [3, 4]]))\n      [1, 2, 3, 4]\n      >>> take(6, flatten([[43, 42], [41, 40], count()]))\n      [43, 42, 41, 40, 0, 1]\n    '
    return chain.from_iterable(xss)

def repeat_func(func, *args, **kwargs):
    if False:
        print('Hello World!')
    "repeat_func(func, *args, **kwargs) -> iterator\n\n    Repeatedly calls `func` with positional arguments `args` and keyword\n    arguments `kwargs`.  If no keyword arguments is given the resulting iterator\n    will be computed using only functions from :mod:`itertools` which are very\n    fast.\n\n    Arguments:\n      func(function):  The function to call.\n      args:  Positional arguments.\n      kwargs:  Keyword arguments.\n\n    Returns:\n      An iterator whoose elements are the results of calling ``func(*args,\n      **kwargs)`` repeatedly.\n\n    Examples:\n      >>> def f(x):\n      ...     x[0] += 1\n      ...     return x[0]\n      >>> i = repeat_func(f, [0])\n      >>> take(2, i)\n      [1, 2]\n      >>> take(2, i)\n      [3, 4]\n      >>> def f(**kwargs):\n      ...     return kwargs.get('x', 43)\n      >>> i = repeat_func(f, x = 42)\n      >>> take(2, i)\n      [42, 42]\n      >>> i = repeat_func(f, 42)\n      >>> take(2, i)\n      Traceback (most recent call last):\n          ...\n      TypeError: f() takes exactly 0 arguments (1 given)\n    "
    if kwargs:
        return starmap(lambda args, kwargs: func(*args, **kwargs), repeat((args, kwargs)))
    else:
        return starmap(func, repeat(args))

def pairwise(iterable):
    if False:
        return 10
    'pairwise(iterable) -> iterator\n\n    Arguments:\n      iterable:  An iterable.\n\n    Returns:\n      An iterator whoose elements are pairs of neighbouring elements of\n      `iterable`.\n\n    Examples:\n      >>> list(pairwise([1, 2, 3, 4]))\n      [(1, 2), (2, 3), (3, 4)]\n      >>> i = starmap(operator.add, pairwise(count()))\n      >>> take(5, i)\n      [1, 3, 5, 7, 9]\n    '
    (a, b) = tee(iterable)
    next(b, None)
    return zip(a, b)

def group(n, iterable, fill_value=None):
    if False:
        for i in range(10):
            print('nop')
    "group(n, iterable, fill_value = None) -> iterator\n\n    Similar to :func:`pwnlib.util.lists.group`, but returns an iterator and uses\n    :mod:`itertools` fast build-in functions.\n\n    Arguments:\n      n(int):  The group size.\n      iterable:  An iterable.\n      fill_value:  The value to fill into the remaining slots of the last group\n        if the `n` does not divide the number of elements in `iterable`.\n\n    Returns:\n      An iterator whoose elements are `n`-tuples of the elements of `iterable`.\n\n    Examples:\n      >>> list(group(2, range(5)))\n      [(0, 1), (2, 3), (4, None)]\n      >>> take(3, group(2, count()))\n      [(0, 1), (2, 3), (4, 5)]\n      >>> [''.join(x) for x in group(3, 'ABCDEFG', 'x')]\n      ['ABC', 'DEF', 'Gxx']\n    "
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fill_value)

def roundrobin(*iterables):
    if False:
        for i in range(10):
            print('nop')
    "roundrobin(*iterables)\n\n    Take elements from `iterables` in a round-robin fashion.\n\n    Arguments:\n      *iterables:  One or more iterables.\n\n    Returns:\n      An iterator whoose elements are taken from `iterables` in a round-robin\n      fashion.\n\n    Examples:\n      >>> ''.join(roundrobin('ABC', 'D', 'EF'))\n      'ADEBFC'\n      >>> ''.join(take(10, roundrobin('ABC', 'DE', repeat('x'))))\n      'ADxBExCxxx'\n    "
    pending = len(iterables)
    nexts = cycle((iter(it) for it in iterables))
    while pending:
        try:
            for nxt in nexts:
                yield next(nxt)
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))

def powerset(iterable, include_empty=True):
    if False:
        return 10
    'powerset(iterable, include_empty = True) -> iterator\n\n    The powerset of an iterable.\n\n    Arguments:\n      iterable:  An iterable.\n      include_empty(bool):  Whether to include the empty set.\n\n    Returns:\n      The powerset of `iterable` as an interator of tuples.\n\n    Examples:\n      >>> list(powerset(range(3)))\n      [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]\n      >>> list(powerset(range(2), include_empty = False))\n      [(0,), (1,), (0, 1)]\n    '
    s = list(iterable)
    i = chain.from_iterable((combinations(s, r) for r in range(len(s) + 1)))
    if not include_empty:
        next(i)
    return i

def unique_everseen(iterable, key=None):
    if False:
        i = 10
        return i + 15
    "unique_everseen(iterable, key = None) -> iterator\n\n    Get unique elements, preserving order. Remember all elements ever seen.  If\n    `key` is not :const:`None` then for each element ``elm`` in `iterable` the\n    element that will be rememberes is ``key(elm)``.  Otherwise ``elm`` is\n    remembered.\n\n    Arguments:\n      iterable:  An iterable.\n      key:  A function to map over each element in `iterable` before remembering\n        it.  Setting to :const:`None` is equivalent to the identity function.\n\n    Returns:\n      An iterator of the unique elements in `iterable`.\n\n    Examples:\n      >>> ''.join(unique_everseen('AAAABBBCCDAABBB'))\n      'ABCD'\n      >>> ''.join(unique_everseen('ABBCcAD', str.lower))\n      'ABCD'\n    "
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

def unique_justseen(iterable, key=None):
    if False:
        return 10
    "unique_everseen(iterable, key = None) -> iterator\n\n    Get unique elements, preserving order. Remember only the elements just seen.\n    If `key` is not :const:`None` then for each element ``elm`` in `iterable`\n    the element that will be rememberes is ``key(elm)``.  Otherwise ``elm`` is\n    remembered.\n\n    Arguments:\n      iterable:  An iterable.\n      key:  A function to map over each element in `iterable` before remembering\n        it.  Setting to :const:`None` is equivalent to the identity function.\n\n    Returns:\n      An iterator of the unique elements in `iterable`.\n\n    Examples:\n      >>> ''.join(unique_justseen('AAAABBBCCDAABBB'))\n      'ABCDAB'\n      >>> ''.join(unique_justseen('ABBCcAD', str.lower))\n      'ABCAD'\n    "
    return map(next, map(operator.itemgetter(1), groupby(iterable, key)))

def unique_window(iterable, window, key=None):
    if False:
        print('Hello World!')
    "unique_everseen(iterable, window, key = None) -> iterator\n\n    Get unique elements, preserving order. Remember only the last `window`\n    elements seen.  If `key` is not :const:`None` then for each element ``elm``\n    in `iterable` the element that will be rememberes is ``key(elm)``.\n    Otherwise ``elm`` is remembered.\n\n    Arguments:\n      iterable:  An iterable.\n      window(int):  The number of elements to remember.\n      key:  A function to map over each element in `iterable` before remembering\n        it.  Setting to :const:`None` is equivalent to the identity function.\n\n    Returns:\n      An iterator of the unique elements in `iterable`.\n\n    Examples:\n      >>> ''.join(unique_window('AAAABBBCCDAABBB', 6))\n      'ABCDA'\n      >>> ''.join(unique_window('ABBCcAD', 5, str.lower))\n      'ABCD'\n      >>> ''.join(unique_window('ABBCcAD', 4, str.lower))\n      'ABCAD'\n    "
    seen = collections.deque(maxlen=window)
    seen_add = seen.append
    if key is None:
        for element in iterable:
            if element not in seen:
                yield element
            seen_add(element)
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                yield element
            seen_add(k)

def iter_except(func, exception):
    if False:
        i = 10
        return i + 15
    'iter_except(func, exception)\n\n    Calls `func` repeatedly until an exception is raised.  Works like the\n    build-in :func:`iter` but uses an exception instead of a sentinel to signal\n    the end.\n\n    Arguments:\n      func(callable): The function to call.\n      exception(Exception):  The exception that signals the end.  Other\n        exceptions will not be caught.\n\n    Returns:\n      An iterator whoose elements are the results of calling ``func()`` until an\n      exception matching `exception` is raised.\n\n    Examples:\n      >>> s = {1, 2, 3}\n      >>> i = iter_except(s.pop, KeyError)\n      >>> next(i)\n      1\n      >>> next(i)\n      2\n      >>> next(i)\n      3\n      >>> next(i)\n      Traceback (most recent call last):\n          ...\n      StopIteration\n    '
    try:
        while True:
            yield func()
    except exception:
        pass

def random_product(*args, **kwargs):
    if False:
        while True:
            i = 10
    'random_product(*args, repeat = 1) -> tuple\n\n    Arguments:\n      args:  One or more iterables\n      repeat(int):  Number of times to repeat `args`.\n\n    Returns:\n      A random element from ``itertools.product(*args, repeat = repeat)``.\n\n    Examples:\n      >>> args = (range(2), range(2))\n      >>> random_product(*args) in {(0, 0), (0, 1), (1, 0), (1, 1)}\n      True\n      >>> args = (range(3), range(3), range(3))\n      >>> random_product(*args, repeat = 2) in product(*args, repeat = 2)\n      True\n    '
    repeat = kwargs.pop('repeat', 1)
    if kwargs != {}:
        raise TypeError('random_product() does not support argument %s' % kwargs.popitem())
    pools = list(map(tuple, args)) * repeat
    return tuple((random.choice(pool) for pool in pools))

def random_permutation(iterable, r=None):
    if False:
        return 10
    'random_product(iterable, r = None) -> tuple\n\n    Arguments:\n      iterable:  An iterable.\n      r(int):  Size of the permutation.  If :const:`None` select all elements in\n        `iterable`.\n\n    Returns:\n      A random element from ``itertools.permutations(iterable, r = r)``.\n\n    Examples:\n      >>> random_permutation(range(2)) in {(0, 1), (1, 0)}\n      True\n      >>> random_permutation(range(10), r = 2) in permutations(range(10), r = 2)\n      True\n    '
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))

def random_combination(iterable, r):
    if False:
        i = 10
        return i + 15
    'random_combination(iterable, r) -> tuple\n\n    Arguments:\n      iterable:  An iterable.\n      r(int):  Size of the combination.\n\n    Returns:\n      A random element from ``itertools.combinations(iterable, r = r)``.\n\n    Examples:\n      >>> random_combination(range(2), 2)\n      (0, 1)\n      >>> random_combination(range(10), r = 2) in combinations(range(10), r = 2)\n      True\n    '
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple((pool[i] for i in indices))

def random_combination_with_replacement(iterable, r):
    if False:
        while True:
            i = 10
    'random_combination(iterable, r) -> tuple\n\n    Arguments:\n      iterable:  An iterable.\n      r(int):  Size of the combination.\n\n    Returns:\n      A random element from ``itertools.combinations_with_replacement(iterable,\n      r = r)``.\n\n    Examples:\n      >>> cs = {(0, 0), (0, 1), (1, 1)}\n      >>> random_combination_with_replacement(range(2), 2) in cs\n      True\n      >>> i = combinations_with_replacement(range(10), r = 2)\n      >>> random_combination_with_replacement(range(10), r = 2) in i\n      True\n    '
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted((random.randrange(n) for i in range(r)))
    return tuple((pool[i] for i in indices))

def lookahead(n, iterable):
    if False:
        for i in range(10):
            print('nop')
    'lookahead(n, iterable) -> object\n\n    Inspects the upcoming element at index `n` without advancing the iterator.\n    Raises ``IndexError`` if `iterable` has too few elements.\n\n    Arguments:\n      n(int):  Index of the element to return.\n      iterable:  An iterable.\n\n    Returns:\n      The element in `iterable` at index `n`.\n\n    Examples:\n      >>> i = count()\n      >>> lookahead(4, i)\n      4\n      >>> next(i)\n      0\n      >>> i = count()\n      >>> nth(4, i)\n      4\n      >>> next(i)\n      5\n      >>> lookahead(4, i)\n      10\n    '
    for value in islice(copy.copy(iterable), n, None):
        return value
    raise IndexError(n)

def lexicographic(alphabet):
    if False:
        for i in range(10):
            print('nop')
    "lexicographic(alphabet) -> iterator\n\n    The words with symbols in `alphabet`, in lexicographic order (determined by\n    the order of `alphabet`).\n\n    Arguments:\n      alphabet:  The alphabet to draw symbols from.\n\n    Returns:\n      An iterator of the words with symbols in `alphabet`, in lexicographic\n      order.\n\n    Example:\n      >>> take(8, map(lambda x: ''.join(x), lexicographic('01')))\n      ['', '0', '1', '00', '01', '10', '11', '000']\n    "
    for n in count():
        for e in product(alphabet, repeat=n):
            yield e

def chained(func):
    if False:
        for i in range(10):
            print('nop')
    'chained(func)\n\n    A decorator chaining the results of `func`.  Useful for generators.\n\n    Arguments:\n      func(function):  The function being decorated.\n\n    Returns:\n      A generator function whoose elements are the concatenation of the return\n      values from ``func(*args, **kwargs)``.\n\n    Example:\n      >>> @chained\n      ... def g():\n      ...     for x in count():\n      ...         yield (x, -x)\n      >>> take(6, g())\n      [0, 0, 1, -1, 2, -2]\n      >>> @chained\n      ... def g2():\n      ...     for x in range(3):\n      ...         yield (x, -x)\n      >>> list(g2())\n      [0, 0, 1, -1, 2, -2]\n    '

    def wrapper(*args, **kwargs):
        if False:
            return 10
        for xs in func(*args, **kwargs):
            for x in xs:
                yield x
    return wrapper

def bruteforce(func, alphabet, length, method='upto', start=None, databag=None):
    if False:
        i = 10
        return i + 15
    "bruteforce(func, alphabet, length, method = 'upto', start = None)\n\n    Bruteforce `func` to return :const:`True`.  `func` should take a string\n    input and return a :func:`bool`.  `func` will be called with strings from\n    `alphabet` until it returns :const:`True` or the search space has been\n    exhausted.\n\n    The argument `start` can be used to split the search space, which is useful\n    if multiple CPU cores are available.\n\n    Arguments:\n      func(function):  The function to bruteforce.\n      alphabet:  The alphabet to draw symbols from.\n      length:  Longest string to try.\n      method:  If 'upto' try strings of length ``1 .. length``, if 'fixed' only\n        try strings of length ``length`` and if 'downfrom' try strings of length\n        ``length .. 1``.\n      start: a tuple ``(i, N)`` which splits the search space up into `N` pieces\n        and starts at piece `i` (1..N). :const:`None` is equivalent to ``(1, 1)``.\n\n    Returns:\n      A string `s` such that ``func(s)`` returns :const:`True` or :const:`None`\n      if the search space was exhausted.\n\n    Example:\n      >>> bruteforce(lambda x: x == 'yes', string.ascii_lowercase, length=5)\n      'yes'\n    "
    if method == 'upto' and length > 1:
        iterator = product(alphabet, repeat=1)
        for i in range(2, length + 1):
            iterator = chain(iterator, product(alphabet, repeat=i))
    elif method == 'downfrom' and length > 1:
        iterator = product(alphabet, repeat=length)
        for i in range(length - 1, 1, -1):
            iterator = chain(iterator, product(alphabet, repeat=i))
    elif method == 'fixed':
        iterator = product(alphabet, repeat=length)
    else:
        raise TypeError('bruteforce(): unknown method')
    if method == 'fixed':
        total_iterations = len(alphabet) ** length
    else:
        total_iterations = len(alphabet) ** (length + 1) // (len(alphabet) - 1) - 1
    if start is not None:
        (i, N) = start
        if i > N:
            raise ValueError('bruteforce(): invalid starting point')
        i -= 1
        chunk_size = total_iterations // N
        rest = total_iterations % N
        starting_point = 0
        for chunk in range(N):
            if chunk >= i:
                break
            if chunk <= rest:
                starting_point += chunk_size + 1
            else:
                starting_point += chunk_size
        if rest >= i:
            chunk_size += 1
        total_iterations = chunk_size
    h = log.waitfor('Bruteforcing')
    cur_iteration = 0
    if start is not None:
        consume(i, iterator)
    for e in iterator:
        cur = ''.join(e)
        cur_iteration += 1
        if cur_iteration % 2000 == 0:
            progress = 100.0 * cur_iteration / total_iterations
            h.status('Trying "%s", %0.3f%%' % (cur, progress))
            if databag:
                databag['current_item'] = cur
                databag['items_done'] = cur_iteration
                databag['items_total'] = total_iterations
        res = func(cur)
        if res:
            h.success('Found key: "%s"' % cur)
            return cur
        if start is not None:
            consume(N - 1, iterator)
    h.failure('No matches found')

def _mbruteforcewrap(func, alphabet, length, method, start, databag):
    if False:
        print('Hello World!')
    oldloglevel = context.log_level
    context.log_level = 'critical'
    res = bruteforce(func, alphabet, length, method=method, start=start, databag=databag)
    context.log_level = oldloglevel
    databag['result'] = res

def mbruteforce(func, alphabet, length, method='upto', start=None, threads=None):
    if False:
        print('Hello World!')
    "mbruteforce(func, alphabet, length, method = 'upto', start = None, threads = None)\n\n    Same functionality as bruteforce(), but multithreaded.\n\n    Arguments:\n      func, alphabet, length, method, start: same as for bruteforce()\n      threads: Amount of threads to spawn, default is the amount of cores.\n\n    Example:\n      >>> mbruteforce(lambda x: x == 'hello', string.ascii_lowercase, length = 10)\n      'hello'\n      >>> mbruteforce(lambda x: x == 'hello', 'hlo', 5, 'downfrom') is None\n      True\n      >>> mbruteforce(lambda x: x == 'no', string.ascii_lowercase, length=2, method='fixed')\n      'no'\n      >>> mbruteforce(lambda x: x == '9999', string.digits, length=4, threads=1, start=(2, 2))\n      '9999'\n    "
    if start is None:
        start = (1, 1)
    if threads is None:
        try:
            threads = multiprocessing.cpu_count()
        except NotImplementedError:
            threads = 1
    h = log.waitfor('MBruteforcing')
    processes = [None] * threads
    shareddata = [None] * threads
    (i2, N2) = start
    totalchunks = threads * N2
    for i in range(threads):
        shareddata[i] = multiprocessing.Manager().dict()
        shareddata[i]['result'] = None
        shareddata[i]['current_item'] = ''
        shareddata[i]['items_done'] = 0
        shareddata[i]['items_total'] = 0
        chunkid = i2 - 1 + i * N2 + 1
        processes[i] = multiprocessing.Process(target=_mbruteforcewrap, args=(func, alphabet, length, method, (chunkid, totalchunks), shareddata[i]))
        processes[i].start()
    done = False
    while not done:
        current_item_list = ','.join(['"%s"' % x['current_item'] for x in shareddata if x is not None])
        items_done = sum([x['items_done'] for x in shareddata if x is not None])
        items_total = sum([x['items_total'] for x in shareddata if x is not None])
        progress = 100.0 * items_done / items_total if items_total != 0 else 0.0
        h.status('Trying %s -- %0.3f%%' % (current_item_list, progress))
        for i in range(threads):
            if processes[i] and processes[i].exitcode is not None:
                res = shareddata[i]['result']
                processes[i].join()
                processes[i] = None
                if res is not None:
                    for i in range(threads):
                        if processes[i] is not None:
                            processes[i].terminate()
                            processes[i].join()
                            processes[i] = None
                    h.success('Found key: "%s"' % res)
                    return res
                if all([x is None for x in processes]):
                    done = True
        time.sleep(0.3)
    h.failure('No matches found')