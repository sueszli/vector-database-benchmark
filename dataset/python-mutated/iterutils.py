""":mod:`itertools` is full of great examples of Python generator
usage. However, there are still some critical gaps. ``iterutils``
fills many of those gaps with featureful, tested, and Pythonic
solutions.

Many of the functions below have two versions, one which
returns an iterator (denoted by the ``*_iter`` naming pattern), and a
shorter-named convenience form that returns a list. Some of the
following are based on examples in itertools docs.
"""
import os
import math
import time
import codecs
import random
import itertools
try:
    from collections.abc import Mapping, Sequence, Set, ItemsView, Iterable
except ImportError:
    from collections import Mapping, Sequence, Set, ItemsView, Iterable
try:
    from .typeutils import make_sentinel
    _UNSET = make_sentinel('_UNSET')
    _REMAP_EXIT = make_sentinel('_REMAP_EXIT')
except ImportError:
    _REMAP_EXIT = object()
    _UNSET = object()
try:
    from future_builtins import filter
    from itertools import izip, izip_longest as zip_longest
    _IS_PY3 = False
except ImportError:
    _IS_PY3 = True
    basestring = (str, bytes)
    unicode = str
    (izip, xrange) = (zip, range)
    from itertools import zip_longest

def is_iterable(obj):
    if False:
        print('Hello World!')
    'Similar in nature to :func:`callable`, ``is_iterable`` returns\n    ``True`` if an object is `iterable`_, ``False`` if not.\n\n    >>> is_iterable([])\n    True\n    >>> is_iterable(object())\n    False\n\n    .. _iterable: https://docs.python.org/2/glossary.html#term-iterable\n    '
    try:
        iter(obj)
    except TypeError:
        return False
    return True

def is_scalar(obj):
    if False:
        i = 10
        return i + 15
    "A near-mirror of :func:`is_iterable`. Returns ``False`` if an\n    object is an iterable container type. Strings are considered\n    scalar as well, because strings are more often treated as whole\n    values as opposed to iterables of 1-character substrings.\n\n    >>> is_scalar(object())\n    True\n    >>> is_scalar(range(10))\n    False\n    >>> is_scalar('hello')\n    True\n    "
    return not is_iterable(obj) or isinstance(obj, basestring)

def is_collection(obj):
    if False:
        while True:
            i = 10
    "The opposite of :func:`is_scalar`.  Returns ``True`` if an object\n    is an iterable other than a string.\n\n    >>> is_collection(object())\n    False\n    >>> is_collection(range(10))\n    True\n    >>> is_collection('hello')\n    False\n    "
    return is_iterable(obj) and (not isinstance(obj, basestring))

def split(src, sep=None, maxsplit=None):
    if False:
        print('Hello World!')
    "Splits an iterable based on a separator. Like :meth:`str.split`,\n    but for all iterables. Returns a list of lists.\n\n    >>> split(['hi', 'hello', None, None, 'sup', None, 'soap', None])\n    [['hi', 'hello'], ['sup'], ['soap']]\n\n    See :func:`split_iter` docs for more info.\n    "
    return list(split_iter(src, sep, maxsplit))

def split_iter(src, sep=None, maxsplit=None):
    if False:
        return 10
    "Splits an iterable based on a separator, *sep*, a max of\n    *maxsplit* times (no max by default). *sep* can be:\n\n      * a single value\n      * an iterable of separators\n      * a single-argument callable that returns True when a separator is\n        encountered\n\n    ``split_iter()`` yields lists of non-separator values. A separator will\n    never appear in the output.\n\n    >>> list(split_iter(['hi', 'hello', None, None, 'sup', None, 'soap', None]))\n    [['hi', 'hello'], ['sup'], ['soap']]\n\n    Note that ``split_iter`` is based on :func:`str.split`, so if\n    *sep* is ``None``, ``split()`` **groups** separators. If empty lists\n    are desired between two contiguous ``None`` values, simply use\n    ``sep=[None]``:\n\n    >>> list(split_iter(['hi', 'hello', None, None, 'sup', None]))\n    [['hi', 'hello'], ['sup']]\n    >>> list(split_iter(['hi', 'hello', None, None, 'sup', None], sep=[None]))\n    [['hi', 'hello'], [], ['sup'], []]\n\n    Using a callable separator:\n\n    >>> falsy_sep = lambda x: not x\n    >>> list(split_iter(['hi', 'hello', None, '', 'sup', False], falsy_sep))\n    [['hi', 'hello'], [], ['sup'], []]\n\n    See :func:`split` for a list-returning version.\n\n    "
    if not is_iterable(src):
        raise TypeError('expected an iterable')
    if maxsplit is not None:
        maxsplit = int(maxsplit)
        if maxsplit == 0:
            yield [src]
            return
    if callable(sep):
        sep_func = sep
    elif not is_scalar(sep):
        sep = frozenset(sep)
        sep_func = lambda x: x in sep
    else:
        sep_func = lambda x: x == sep
    cur_group = []
    split_count = 0
    for s in src:
        if maxsplit is not None and split_count >= maxsplit:
            sep_func = lambda x: False
        if sep_func(s):
            if sep is None and (not cur_group):
                continue
            split_count += 1
            yield cur_group
            cur_group = []
        else:
            cur_group.append(s)
    if cur_group or sep is not None:
        yield cur_group
    return

def lstrip(iterable, strip_value=None):
    if False:
        return 10
    "Strips values from the beginning of an iterable. Stripped items will\n    match the value of the argument strip_value. Functionality is analogous\n    to that of the method str.lstrip. Returns a list.\n\n    >>> lstrip(['Foo', 'Bar', 'Bam'], 'Foo')\n    ['Bar', 'Bam']\n\n    "
    return list(lstrip_iter(iterable, strip_value))

def lstrip_iter(iterable, strip_value=None):
    if False:
        for i in range(10):
            print('nop')
    "Strips values from the beginning of an iterable. Stripped items will\n    match the value of the argument strip_value. Functionality is analogous\n    to that of the method str.lstrip. Returns a generator.\n\n    >>> list(lstrip_iter(['Foo', 'Bar', 'Bam'], 'Foo'))\n    ['Bar', 'Bam']\n\n    "
    iterator = iter(iterable)
    for i in iterator:
        if i != strip_value:
            yield i
            break
    for i in iterator:
        yield i

def rstrip(iterable, strip_value=None):
    if False:
        return 10
    "Strips values from the end of an iterable. Stripped items will\n    match the value of the argument strip_value. Functionality is analogous\n    to that of the method str.rstrip. Returns a list.\n\n    >>> rstrip(['Foo', 'Bar', 'Bam'], 'Bam')\n    ['Foo', 'Bar']\n\n    "
    return list(rstrip_iter(iterable, strip_value))

def rstrip_iter(iterable, strip_value=None):
    if False:
        print('Hello World!')
    "Strips values from the end of an iterable. Stripped items will\n    match the value of the argument strip_value. Functionality is analogous\n    to that of the method str.rstrip. Returns a generator.\n\n    >>> list(rstrip_iter(['Foo', 'Bar', 'Bam'], 'Bam'))\n    ['Foo', 'Bar']\n\n    "
    iterator = iter(iterable)
    for i in iterator:
        if i == strip_value:
            cache = list()
            cache.append(i)
            broken = False
            for i in iterator:
                if i == strip_value:
                    cache.append(i)
                else:
                    broken = True
                    break
            if not broken:
                return
            for t in cache:
                yield t
        yield i

def strip(iterable, strip_value=None):
    if False:
        print('Hello World!')
    "Strips values from the beginning and end of an iterable. Stripped items\n    will match the value of the argument strip_value. Functionality is\n    analogous to that of the method str.strip. Returns a list.\n\n    >>> strip(['Fu', 'Foo', 'Bar', 'Bam', 'Fu'], 'Fu')\n    ['Foo', 'Bar', 'Bam']\n\n    "
    return list(strip_iter(iterable, strip_value))

def strip_iter(iterable, strip_value=None):
    if False:
        i = 10
        return i + 15
    "Strips values from the beginning and end of an iterable. Stripped items\n    will match the value of the argument strip_value. Functionality is\n    analogous to that of the method str.strip. Returns a generator.\n\n    >>> list(strip_iter(['Fu', 'Foo', 'Bar', 'Bam', 'Fu'], 'Fu'))\n    ['Foo', 'Bar', 'Bam']\n\n    "
    return rstrip_iter(lstrip_iter(iterable, strip_value), strip_value)

def chunked(src, size, count=None, **kw):
    if False:
        while True:
            i = 10
    'Returns a list of *count* chunks, each with *size* elements,\n    generated from iterable *src*. If *src* is not evenly divisible by\n    *size*, the final chunk will have fewer than *size* elements.\n    Provide the *fill* keyword argument to provide a pad value and\n    enable padding, otherwise no padding will take place.\n\n    >>> chunked(range(10), 3)\n    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]\n    >>> chunked(range(10), 3, fill=None)\n    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, None, None]]\n    >>> chunked(range(10), 3, count=2)\n    [[0, 1, 2], [3, 4, 5]]\n\n    See :func:`chunked_iter` for more info.\n    '
    chunk_iter = chunked_iter(src, size, **kw)
    if count is None:
        return list(chunk_iter)
    else:
        return list(itertools.islice(chunk_iter, count))

def _validate_positive_int(value, name, strictly_positive=True):
    if False:
        for i in range(10):
            print('nop')
    value = int(value)
    if value < 0 or (strictly_positive and value == 0):
        raise ValueError('expected a positive integer ' + name)
    return value

def chunked_iter(src, size, **kw):
    if False:
        i = 10
        return i + 15
    'Generates *size*-sized chunks from *src* iterable. Unless the\n    optional *fill* keyword argument is provided, iterables not evenly\n    divisible by *size* will have a final chunk that is smaller than\n    *size*.\n\n    >>> list(chunked_iter(range(10), 3))\n    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]\n    >>> list(chunked_iter(range(10), 3, fill=None))\n    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, None, None]]\n\n    Note that ``fill=None`` in fact uses ``None`` as the fill value.\n    '
    if not is_iterable(src):
        raise TypeError('expected an iterable')
    size = _validate_positive_int(size, 'chunk size')
    do_fill = True
    try:
        fill_val = kw.pop('fill')
    except KeyError:
        do_fill = False
        fill_val = None
    if kw:
        raise ValueError('got unexpected keyword arguments: %r' % kw.keys())
    if not src:
        return
    postprocess = lambda chk: chk
    if isinstance(src, basestring):
        postprocess = lambda chk, _sep=type(src)(): _sep.join(chk)
        if _IS_PY3 and isinstance(src, bytes):
            postprocess = lambda chk: bytes(chk)
    src_iter = iter(src)
    while True:
        cur_chunk = list(itertools.islice(src_iter, size))
        if not cur_chunk:
            break
        lc = len(cur_chunk)
        if lc < size and do_fill:
            cur_chunk[lc:] = [fill_val] * (size - lc)
        yield postprocess(cur_chunk)
    return

def chunk_ranges(input_size, chunk_size, input_offset=0, overlap_size=0, align=False):
    if False:
        while True:
            i = 10
    'Generates *chunk_size*-sized chunk ranges for an input with length *input_size*.\n    Optionally, a start of the input can be set via *input_offset*, and\n    and overlap between the chunks may be specified via *overlap_size*.\n    Also, if *align* is set to *True*, any items with *i % (chunk_size-overlap_size) == 0*\n    are always at the beginning of the chunk.\n\n    Returns an iterator of (start, end) tuples, one tuple per chunk.\n\n    >>> list(chunk_ranges(input_offset=10, input_size=10, chunk_size=5))\n    [(10, 15), (15, 20)]\n    >>> list(chunk_ranges(input_offset=10, input_size=10, chunk_size=5, overlap_size=1))\n    [(10, 15), (14, 19), (18, 20)]\n    >>> list(chunk_ranges(input_offset=10, input_size=10, chunk_size=5, overlap_size=2))\n    [(10, 15), (13, 18), (16, 20)]\n\n    >>> list(chunk_ranges(input_offset=4, input_size=15, chunk_size=5, align=False))\n    [(4, 9), (9, 14), (14, 19)]\n    >>> list(chunk_ranges(input_offset=4, input_size=15, chunk_size=5, align=True))\n    [(4, 5), (5, 10), (10, 15), (15, 19)]\n\n    >>> list(chunk_ranges(input_offset=2, input_size=15, chunk_size=5, overlap_size=1, align=False))\n    [(2, 7), (6, 11), (10, 15), (14, 17)]\n    >>> list(chunk_ranges(input_offset=2, input_size=15, chunk_size=5, overlap_size=1, align=True))\n    [(2, 5), (4, 9), (8, 13), (12, 17)]\n    >>> list(chunk_ranges(input_offset=3, input_size=15, chunk_size=5, overlap_size=1, align=True))\n    [(3, 5), (4, 9), (8, 13), (12, 17), (16, 18)]\n    '
    input_size = _validate_positive_int(input_size, 'input_size', strictly_positive=False)
    chunk_size = _validate_positive_int(chunk_size, 'chunk_size')
    input_offset = _validate_positive_int(input_offset, 'input_offset', strictly_positive=False)
    overlap_size = _validate_positive_int(overlap_size, 'overlap_size', strictly_positive=False)
    input_stop = input_offset + input_size
    if align:
        initial_chunk_len = chunk_size - input_offset % (chunk_size - overlap_size)
        if initial_chunk_len != overlap_size:
            yield (input_offset, min(input_offset + initial_chunk_len, input_stop))
            if input_offset + initial_chunk_len >= input_stop:
                return
            input_offset = input_offset + initial_chunk_len - overlap_size
    for i in range(input_offset, input_stop, chunk_size - overlap_size):
        yield (i, min(i + chunk_size, input_stop))
        if i + chunk_size >= input_stop:
            return

def pairwise(src, end=_UNSET):
    if False:
        i = 10
        return i + 15
    'Convenience function for calling :func:`windowed` on *src*, with\n    *size* set to 2.\n\n    >>> pairwise(range(5))\n    [(0, 1), (1, 2), (2, 3), (3, 4)]\n    >>> pairwise([])\n    []\n\n    Unless *end* is set, the number of pairs is always one less than \n    the number of elements in the iterable passed in, except on an empty input, \n    which will return an empty list.\n\n    With *end* set, a number of pairs equal to the length of *src* is returned,\n    with the last item of the last pair being equal to *end*.\n\n    >>> list(pairwise(range(3), end=None))\n    [(0, 1), (1, 2), (2, None)]\n\n    This way, *end* values can be useful as sentinels to signal the end of the iterable.\n    '
    return windowed(src, 2, fill=end)

def pairwise_iter(src, end=_UNSET):
    if False:
        return 10
    'Convenience function for calling :func:`windowed_iter` on *src*,\n    with *size* set to 2.\n\n    >>> list(pairwise_iter(range(5)))\n    [(0, 1), (1, 2), (2, 3), (3, 4)]\n    >>> list(pairwise_iter([]))\n    []\n\n    Unless *end* is set, the number of pairs is always one less \n    than the number of elements in the iterable passed in, \n    or zero, when *src* is empty.\n\n    With *end* set, a number of pairs equal to the length of *src* is returned,\n    with the last item of the last pair being equal to *end*. \n\n    >>> list(pairwise_iter(range(3), end=None))\n    [(0, 1), (1, 2), (2, None)]    \n\n    This way, *end* values can be useful as sentinels to signal the end\n    of the iterable. For infinite iterators, setting *end* has no effect.\n    '
    return windowed_iter(src, 2, fill=end)

def windowed(src, size, fill=_UNSET):
    if False:
        i = 10
        return i + 15
    'Returns tuples with exactly length *size*. If *fill* is unset \n    and the iterable is too short to make a window of length *size*, \n    no tuples are returned. See :func:`windowed_iter` for more.\n    '
    return list(windowed_iter(src, size, fill=fill))

def windowed_iter(src, size, fill=_UNSET):
    if False:
        return 10
    'Returns tuples with length *size* which represent a sliding\n    window over iterable *src*.\n\n    >>> list(windowed_iter(range(7), 3))\n    [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6)]\n\n    If *fill* is unset, and the iterable is too short to make a window \n    of length *size*, then no window tuples are returned.\n\n    >>> list(windowed_iter(range(3), 5))\n    []\n\n    With *fill* set, the iterator always yields a number of windows\n    equal to the length of the *src* iterable.\n    \n    >>> windowed(range(4), 3, fill=None)\n    [(0, 1, 2), (1, 2, 3), (2, 3, None), (3, None, None)]\n\n    This way, *fill* values can be useful to signal the end of the iterable.\n    For infinite iterators, setting *fill* has no effect.\n    '
    tees = itertools.tee(src, size)
    if fill is _UNSET:
        try:
            for (i, t) in enumerate(tees):
                for _ in range(i):
                    next(t)
        except StopIteration:
            return zip([])
        return zip(*tees)
    for (i, t) in enumerate(tees):
        for _ in range(i):
            try:
                next(t)
            except StopIteration:
                continue
    return zip_longest(*tees, fillvalue=fill)

def xfrange(stop, start=None, step=1.0):
    if False:
        return 10
    'Same as :func:`frange`, but generator-based instead of returning a\n    list.\n\n    >>> tuple(xfrange(1, 3, step=0.75))\n    (1.0, 1.75, 2.5)\n\n    See :func:`frange` for more details.\n    '
    if not step:
        raise ValueError('step must be non-zero')
    if start is None:
        (start, stop) = (0.0, stop * 1.0)
    else:
        (stop, start) = (start * 1.0, stop * 1.0)
    cur = start
    while cur < stop:
        yield cur
        cur += step

def frange(stop, start=None, step=1.0):
    if False:
        for i in range(10):
            print('nop')
    'A :func:`range` clone for float-based ranges.\n\n    >>> frange(5)\n    [0.0, 1.0, 2.0, 3.0, 4.0]\n    >>> frange(6, step=1.25)\n    [0.0, 1.25, 2.5, 3.75, 5.0]\n    >>> frange(100.5, 101.5, 0.25)\n    [100.5, 100.75, 101.0, 101.25]\n    >>> frange(5, 0)\n    []\n    >>> frange(5, 0, step=-1.25)\n    [5.0, 3.75, 2.5, 1.25]\n    '
    if not step:
        raise ValueError('step must be non-zero')
    if start is None:
        (start, stop) = (0.0, stop * 1.0)
    else:
        (stop, start) = (start * 1.0, stop * 1.0)
    count = int(math.ceil((stop - start) / step))
    ret = [None] * count
    if not ret:
        return ret
    ret[0] = start
    for i in xrange(1, count):
        ret[i] = ret[i - 1] + step
    return ret

def backoff(start, stop, count=None, factor=2.0, jitter=False):
    if False:
        while True:
            i = 10
    "Returns a list of geometrically-increasing floating-point numbers,\n    suitable for usage with `exponential backoff`_. Exactly like\n    :func:`backoff_iter`, but without the ``'repeat'`` option for\n    *count*. See :func:`backoff_iter` for more details.\n\n    .. _exponential backoff: https://en.wikipedia.org/wiki/Exponential_backoff\n\n    >>> backoff(1, 10)\n    [1.0, 2.0, 4.0, 8.0, 10.0]\n    "
    if count == 'repeat':
        raise ValueError("'repeat' supported in backoff_iter, not backoff")
    return list(backoff_iter(start, stop, count=count, factor=factor, jitter=jitter))

def backoff_iter(start, stop, count=None, factor=2.0, jitter=False):
    if False:
        return 10
    "Generates a sequence of geometrically-increasing floats, suitable\n    for usage with `exponential backoff`_. Starts with *start*,\n    increasing by *factor* until *stop* is reached, optionally\n    stopping iteration once *count* numbers are yielded. *factor*\n    defaults to 2. In general retrying with properly-configured\n    backoff creates a better-behaved component for a larger service\n    ecosystem.\n\n    .. _exponential backoff: https://en.wikipedia.org/wiki/Exponential_backoff\n\n    >>> list(backoff_iter(1.0, 10.0, count=5))\n    [1.0, 2.0, 4.0, 8.0, 10.0]\n    >>> list(backoff_iter(1.0, 10.0, count=8))\n    [1.0, 2.0, 4.0, 8.0, 10.0, 10.0, 10.0, 10.0]\n    >>> list(backoff_iter(0.25, 100.0, factor=10))\n    [0.25, 2.5, 25.0, 100.0]\n\n    A simplified usage example:\n\n    .. code-block:: python\n\n      for timeout in backoff_iter(0.25, 5.0):\n          try:\n              res = network_call()\n              break\n          except Exception as e:\n              log(e)\n              time.sleep(timeout)\n\n    An enhancement for large-scale systems would be to add variation,\n    or *jitter*, to timeout values. This is done to avoid a thundering\n    herd on the receiving end of the network call.\n\n    Finally, for *count*, the special value ``'repeat'`` can be passed to\n    continue yielding indefinitely.\n\n    Args:\n\n        start (float): Positive number for baseline.\n        stop (float): Positive number for maximum.\n        count (int): Number of steps before stopping\n            iteration. Defaults to the number of steps between *start* and\n            *stop*. Pass the string, `'repeat'`, to continue iteration\n            indefinitely.\n        factor (float): Rate of exponential increase. Defaults to `2.0`,\n            e.g., `[1, 2, 4, 8, 16]`.\n        jitter (float): A factor between `-1.0` and `1.0`, used to\n            uniformly randomize and thus spread out timeouts in a distributed\n            system, avoiding rhythm effects. Positive values use the base\n            backoff curve as a maximum, negative values use the curve as a\n            minimum. Set to 1.0 or `True` for a jitter approximating\n            Ethernet's time-tested backoff solution. Defaults to `False`.\n\n    "
    start = float(start)
    stop = float(stop)
    factor = float(factor)
    if start < 0.0:
        raise ValueError('expected start >= 0, not %r' % start)
    if factor < 1.0:
        raise ValueError('expected factor >= 1.0, not %r' % factor)
    if stop == 0.0:
        raise ValueError('expected stop >= 0')
    if stop < start:
        raise ValueError('expected stop >= start, not %r' % stop)
    if count is None:
        denom = start if start else 1
        count = 1 + math.ceil(math.log(stop / denom, factor))
        count = count if start else count + 1
    if count != 'repeat' and count < 0:
        raise ValueError('count must be positive or "repeat", not %r' % count)
    if jitter:
        jitter = float(jitter)
        if not -1.0 <= jitter <= 1.0:
            raise ValueError('expected jitter -1 <= j <= 1, not: %r' % jitter)
    (cur, i) = (start, 0)
    while count == 'repeat' or i < count:
        if not jitter:
            cur_ret = cur
        elif jitter:
            cur_ret = cur - cur * jitter * random.random()
        yield cur_ret
        i += 1
        if cur == 0:
            cur = 1
        elif cur < stop:
            cur *= factor
        if cur > stop:
            cur = stop
    return

def bucketize(src, key=bool, value_transform=None, key_filter=None):
    if False:
        return 10
    "Group values in the *src* iterable by the value returned by *key*.\n\n    >>> bucketize(range(5))\n    {False: [0], True: [1, 2, 3, 4]}\n    >>> is_odd = lambda x: x % 2 == 1\n    >>> bucketize(range(5), is_odd)\n    {False: [0, 2, 4], True: [1, 3]}\n\n    *key* is :class:`bool` by default, but can either be a callable or a string or a list\n    if it is a string, it is the name of the attribute on which to bucketize objects.\n\n    >>> bucketize([1+1j, 2+2j, 1, 2], key='real')\n    {1.0: [(1+1j), 1], 2.0: [(2+2j), 2]}\n\n    if *key* is a list, it contains the buckets where to put each object\n\n    >>> bucketize([1,2,365,4,98],key=[0,1,2,0,2])\n    {0: [1, 4], 1: [2], 2: [365, 98]}\n\n\n    Value lists are not deduplicated:\n\n    >>> bucketize([None, None, None, 'hello'])\n    {False: [None, None, None], True: ['hello']}\n\n    Bucketize into more than 3 groups\n\n    >>> bucketize(range(10), lambda x: x % 3)\n    {0: [0, 3, 6, 9], 1: [1, 4, 7], 2: [2, 5, 8]}\n\n    ``bucketize`` has a couple of advanced options useful in certain\n    cases.  *value_transform* can be used to modify values as they are\n    added to buckets, and *key_filter* will allow excluding certain\n    buckets from being collected.\n\n    >>> bucketize(range(5), value_transform=lambda x: x*x)\n    {False: [0], True: [1, 4, 9, 16]}\n\n    >>> bucketize(range(10), key=lambda x: x % 3, key_filter=lambda k: k % 3 != 1)\n    {0: [0, 3, 6, 9], 2: [2, 5, 8]}\n\n    Note in some of these examples there were at most two keys, ``True`` and\n    ``False``, and each key present has a list with at least one\n    item. See :func:`partition` for a version specialized for binary\n    use cases.\n\n    "
    if not is_iterable(src):
        raise TypeError('expected an iterable')
    elif isinstance(key, list):
        if len(key) != len(src):
            raise ValueError('key and src have to be the same length')
        src = zip(key, src)
    if isinstance(key, basestring):
        key_func = lambda x: getattr(x, key, x)
    elif callable(key):
        key_func = key
    elif isinstance(key, list):
        key_func = lambda x: x[0]
    else:
        raise TypeError('expected key to be callable or a string or a list')
    if value_transform is None:
        value_transform = lambda x: x
    if not callable(value_transform):
        raise TypeError('expected callable value transform function')
    if isinstance(key, list):
        f = value_transform
        value_transform = lambda x: f(x[1])
    ret = {}
    for val in src:
        key_of_val = key_func(val)
        if key_filter is None or key_filter(key_of_val):
            ret.setdefault(key_of_val, []).append(value_transform(val))
    return ret

def partition(src, key=bool):
    if False:
        print('Hello World!')
    "No relation to :meth:`str.partition`, ``partition`` is like\n    :func:`bucketize`, but for added convenience returns a tuple of\n    ``(truthy_values, falsy_values)``.\n\n    >>> nonempty, empty = partition(['', '', 'hi', '', 'bye'])\n    >>> nonempty\n    ['hi', 'bye']\n\n    *key* defaults to :class:`bool`, but can be carefully overridden to\n    use either a function that returns either ``True`` or ``False`` or\n    a string name of the attribute on which to partition objects.\n\n    >>> import string\n    >>> is_digit = lambda x: x in string.digits\n    >>> decimal_digits, hexletters = partition(string.hexdigits, is_digit)\n    >>> ''.join(decimal_digits), ''.join(hexletters)\n    ('0123456789', 'abcdefABCDEF')\n    "
    bucketized = bucketize(src, key)
    return (bucketized.get(True, []), bucketized.get(False, []))

def unique(src, key=None):
    if False:
        return 10
    "``unique()`` returns a list of unique values, as determined by\n    *key*, in the order they first appeared in the input iterable,\n    *src*.\n\n    >>> ones_n_zeros = '11010110001010010101010'\n    >>> ''.join(unique(ones_n_zeros))\n    '10'\n\n    See :func:`unique_iter` docs for more details.\n    "
    return list(unique_iter(src, key))

def unique_iter(src, key=None):
    if False:
        for i in range(10):
            print('nop')
    "Yield unique elements from the iterable, *src*, based on *key*,\n    in the order in which they first appeared in *src*.\n\n    >>> repetitious = [1, 2, 3] * 10\n    >>> list(unique_iter(repetitious))\n    [1, 2, 3]\n\n    By default, *key* is the object itself, but *key* can either be a\n    callable or, for convenience, a string name of the attribute on\n    which to uniqueify objects, falling back on identity when the\n    attribute is not present.\n\n    >>> pleasantries = ['hi', 'hello', 'ok', 'bye', 'yes']\n    >>> list(unique_iter(pleasantries, key=lambda x: len(x)))\n    ['hi', 'hello', 'bye']\n    "
    if not is_iterable(src):
        raise TypeError('expected an iterable, not %r' % type(src))
    if key is None:
        key_func = lambda x: x
    elif callable(key):
        key_func = key
    elif isinstance(key, basestring):
        key_func = lambda x: getattr(x, key, x)
    else:
        raise TypeError('"key" expected a string or callable, not %r' % key)
    seen = set()
    for i in src:
        k = key_func(i)
        if k not in seen:
            seen.add(k)
            yield i
    return

def redundant(src, key=None, groups=False):
    if False:
        while True:
            i = 10
    "The complement of :func:`unique()`.\n\n    By default returns non-unique/duplicate values as a list of the\n    *first* redundant value in *src*. Pass ``groups=True`` to get\n    groups of all values with redundancies, ordered by position of the\n    first redundant value. This is useful in conjunction with some\n    normalizing *key* function.\n\n    >>> redundant([1, 2, 3, 4])\n    []\n    >>> redundant([1, 2, 3, 2, 3, 3, 4])\n    [2, 3]\n    >>> redundant([1, 2, 3, 2, 3, 3, 4], groups=True)\n    [[2, 2], [3, 3, 3]]\n\n    An example using a *key* function to do case-insensitive\n    redundancy detection.\n\n    >>> redundant(['hi', 'Hi', 'HI', 'hello'], key=str.lower)\n    ['Hi']\n    >>> redundant(['hi', 'Hi', 'HI', 'hello'], groups=True, key=str.lower)\n    [['hi', 'Hi', 'HI']]\n\n    *key* should also be used when the values in *src* are not hashable.\n\n    .. note::\n\n       This output of this function is designed for reporting\n       duplicates in contexts when a unique input is desired. Due to\n       the grouped return type, there is no streaming equivalent of\n       this function for the time being.\n\n    "
    if key is None:
        pass
    elif callable(key):
        key_func = key
    elif isinstance(key, basestring):
        key_func = lambda x: getattr(x, key, x)
    else:
        raise TypeError('"key" expected a string or callable, not %r' % key)
    seen = {}
    redundant_order = []
    redundant_groups = {}
    for i in src:
        k = key_func(i) if key else i
        if k not in seen:
            seen[k] = i
        elif k in redundant_groups:
            if groups:
                redundant_groups[k].append(i)
        else:
            redundant_order.append(k)
            redundant_groups[k] = [seen[k], i]
    if not groups:
        ret = [redundant_groups[k][1] for k in redundant_order]
    else:
        ret = [redundant_groups[k] for k in redundant_order]
    return ret

def one(src, default=None, key=None):
    if False:
        print('Hello World!')
    "Along the same lines as builtins, :func:`all` and :func:`any`, and\n    similar to :func:`first`, ``one()`` returns the single object in\n    the given iterable *src* that evaluates to ``True``, as determined\n    by callable *key*. If unset, *key* defaults to :class:`bool`. If\n    no such objects are found, *default* is returned. If *default* is\n    not passed, ``None`` is returned.\n\n    If *src* has more than one object that evaluates to ``True``, or\n    if there is no object that fulfills such condition, return\n    *default*. It's like an `XOR`_ over an iterable.\n\n    >>> one((True, False, False))\n    True\n    >>> one((True, False, True))\n    >>> one((0, 0, 'a'))\n    'a'\n    >>> one((0, False, None))\n    >>> one((True, True), default=False)\n    False\n    >>> bool(one(('', 1)))\n    True\n    >>> one((10, 20, 30, 42), key=lambda i: i > 40)\n    42\n\n    See `Martín Gaitán's original repo`_ for further use cases.\n\n    .. _Martín Gaitán's original repo: https://github.com/mgaitan/one\n    .. _XOR: https://en.wikipedia.org/wiki/Exclusive_or\n\n    "
    ones = list(itertools.islice(filter(key, src), 2))
    return ones[0] if len(ones) == 1 else default

def first(iterable, default=None, key=None):
    if False:
        while True:
            i = 10
    "Return first element of *iterable* that evaluates to ``True``, else\n    return ``None`` or optional *default*. Similar to :func:`one`.\n\n    >>> first([0, False, None, [], (), 42])\n    42\n    >>> first([0, False, None, [], ()]) is None\n    True\n    >>> first([0, False, None, [], ()], default='ohai')\n    'ohai'\n    >>> import re\n    >>> m = first(re.match(regex, 'abc') for regex in ['b.*', 'a(.*)'])\n    >>> m.group(1)\n    'bc'\n\n    The optional *key* argument specifies a one-argument predicate function\n    like that used for *filter()*.  The *key* argument, if supplied, should be\n    in keyword form. For example, finding the first even number in an iterable:\n\n    >>> first([1, 1, 3, 4, 5], key=lambda x: x % 2 == 0)\n    4\n\n    Contributed by Hynek Schlawack, author of `the original standalone module`_.\n\n    .. _the original standalone module: https://github.com/hynek/first\n    "
    return next(filter(key, iterable), default)

def flatten_iter(iterable):
    if False:
        print('Hello World!')
    '``flatten_iter()`` yields all the elements from *iterable* while\n    collapsing any nested iterables.\n\n    >>> nested = [[1, 2], [[3], [4, 5]]]\n    >>> list(flatten_iter(nested))\n    [1, 2, 3, 4, 5]\n    '
    for item in iterable:
        if isinstance(item, Iterable) and (not isinstance(item, basestring)):
            for subitem in flatten_iter(item):
                yield subitem
        else:
            yield item

def flatten(iterable):
    if False:
        while True:
            i = 10
    '``flatten()`` returns a collapsed list of all the elements from\n    *iterable* while collapsing any nested iterables.\n\n    >>> nested = [[1, 2], [[3], [4, 5]]]\n    >>> flatten(nested)\n    [1, 2, 3, 4, 5]\n    '
    return list(flatten_iter(iterable))

def same(iterable, ref=_UNSET):
    if False:
        i = 10
        return i + 15
    "``same()`` returns ``True`` when all values in *iterable* are\n    equal to one another, or optionally a reference value,\n    *ref*. Similar to :func:`all` and :func:`any` in that it evaluates\n    an iterable and returns a :class:`bool`. ``same()`` returns\n    ``True`` for empty iterables.\n\n    >>> same([])\n    True\n    >>> same([1])\n    True\n    >>> same(['a', 'a', 'a'])\n    True\n    >>> same(range(20))\n    False\n    >>> same([[], []])\n    True\n    >>> same([[], []], ref='test')\n    False\n\n    "
    iterator = iter(iterable)
    if ref is _UNSET:
        ref = next(iterator, ref)
    return all((val == ref for val in iterator))

def default_visit(path, key, value):
    if False:
        while True:
            i = 10
    return (key, value)
_orig_default_visit = default_visit

def default_enter(path, key, value):
    if False:
        return 10
    if isinstance(value, basestring):
        return (value, False)
    elif isinstance(value, Mapping):
        return (value.__class__(), ItemsView(value))
    elif isinstance(value, Sequence):
        return (value.__class__(), enumerate(value))
    elif isinstance(value, Set):
        return (value.__class__(), enumerate(value))
    else:
        return (value, False)

def default_exit(path, key, old_parent, new_parent, new_items):
    if False:
        for i in range(10):
            print('nop')
    ret = new_parent
    if isinstance(new_parent, Mapping):
        new_parent.update(new_items)
    elif isinstance(new_parent, Sequence):
        vals = [v for (i, v) in new_items]
        try:
            new_parent.extend(vals)
        except AttributeError:
            ret = new_parent.__class__(vals)
    elif isinstance(new_parent, Set):
        vals = [v for (i, v) in new_items]
        try:
            new_parent.update(vals)
        except AttributeError:
            ret = new_parent.__class__(vals)
    else:
        raise RuntimeError('unexpected iterable type: %r' % type(new_parent))
    return ret

def remap(root, visit=default_visit, enter=default_enter, exit=default_exit, **kwargs):
    if False:
        i = 10
        return i + 15
    'The remap ("recursive map") function is used to traverse and\n    transform nested structures. Lists, tuples, sets, and dictionaries\n    are just a few of the data structures nested into heterogeneous\n    tree-like structures that are so common in programming.\n    Unfortunately, Python\'s built-in ways to manipulate collections\n    are almost all flat. List comprehensions may be fast and succinct,\n    but they do not recurse, making it tedious to apply quick changes\n    or complex transforms to real-world data.\n\n    remap goes where list comprehensions cannot.\n\n    Here\'s an example of removing all Nones from some data:\n\n    >>> from pprint import pprint\n    >>> reviews = {\'Star Trek\': {\'TNG\': 10, \'DS9\': 8.5, \'ENT\': None},\n    ...            \'Babylon 5\': 6, \'Dr. Who\': None}\n    >>> pprint(remap(reviews, lambda p, k, v: v is not None))\n    {\'Babylon 5\': 6, \'Star Trek\': {\'DS9\': 8.5, \'TNG\': 10}}\n\n    Notice how both Nones have been removed despite the nesting in the\n    dictionary. Not bad for a one-liner, and that\'s just the beginning.\n    See `this remap cookbook`_ for more delicious recipes.\n\n    .. _this remap cookbook: http://sedimental.org/remap.html\n\n    remap takes four main arguments: the object to traverse and three\n    optional callables which determine how the remapped object will be\n    created.\n\n    Args:\n\n        root: The target object to traverse. By default, remap\n            supports iterables like :class:`list`, :class:`tuple`,\n            :class:`dict`, and :class:`set`, but any object traversable by\n            *enter* will work.\n        visit (callable): This function is called on every item in\n            *root*. It must accept three positional arguments, *path*,\n            *key*, and *value*. *path* is simply a tuple of parents\'\n            keys. *visit* should return the new key-value pair. It may\n            also return ``True`` as shorthand to keep the old item\n            unmodified, or ``False`` to drop the item from the new\n            structure. *visit* is called after *enter*, on the new parent.\n\n            The *visit* function is called for every item in root,\n            including duplicate items. For traversable values, it is\n            called on the new parent object, after all its children\n            have been visited. The default visit behavior simply\n            returns the key-value pair unmodified.\n        enter (callable): This function controls which items in *root*\n            are traversed. It accepts the same arguments as *visit*: the\n            path, the key, and the value of the current item. It returns a\n            pair of the blank new parent, and an iterator over the items\n            which should be visited. If ``False`` is returned instead of\n            an iterator, the value will not be traversed.\n\n            The *enter* function is only called once per unique value. The\n            default enter behavior support mappings, sequences, and\n            sets. Strings and all other iterables will not be traversed.\n        exit (callable): This function determines how to handle items\n            once they have been visited. It gets the same three\n            arguments as the other functions -- *path*, *key*, *value*\n            -- plus two more: the blank new parent object returned\n            from *enter*, and a list of the new items, as remapped by\n            *visit*.\n\n            Like *enter*, the *exit* function is only called once per\n            unique value. The default exit behavior is to simply add\n            all new items to the new parent, e.g., using\n            :meth:`list.extend` and :meth:`dict.update` to add to the\n            new parent. Immutable objects, such as a :class:`tuple` or\n            :class:`namedtuple`, must be recreated from scratch, but\n            use the same type as the new parent passed back from the\n            *enter* function.\n        reraise_visit (bool): A pragmatic convenience for the *visit*\n            callable. When set to ``False``, remap ignores any errors\n            raised by the *visit* callback. Items causing exceptions\n            are kept. See examples for more details.\n\n    remap is designed to cover the majority of cases with just the\n    *visit* callable. While passing in multiple callables is very\n    empowering, remap is designed so very few cases should require\n    passing more than one function.\n\n    When passing *enter* and *exit*, it\'s common and easiest to build\n    on the default behavior. Simply add ``from boltons.iterutils import\n    default_enter`` (or ``default_exit``), and have your enter/exit\n    function call the default behavior before or after your custom\n    logic. See `this example`_.\n\n    Duplicate and self-referential objects (aka reference loops) are\n    automatically handled internally, `as shown here`_.\n\n    .. _this example: http://sedimental.org/remap.html#sort_all_lists\n    .. _as shown here: http://sedimental.org/remap.html#corner_cases\n\n    '
    if not callable(visit):
        raise TypeError('visit expected callable, not: %r' % visit)
    if not callable(enter):
        raise TypeError('enter expected callable, not: %r' % enter)
    if not callable(exit):
        raise TypeError('exit expected callable, not: %r' % exit)
    reraise_visit = kwargs.pop('reraise_visit', True)
    if kwargs:
        raise TypeError('unexpected keyword arguments: %r' % kwargs.keys())
    (path, registry, stack) = ((), {}, [(None, root)])
    new_items_stack = []
    while stack:
        (key, value) = stack.pop()
        id_value = id(value)
        if key is _REMAP_EXIT:
            (key, new_parent, old_parent) = value
            id_value = id(old_parent)
            (path, new_items) = new_items_stack.pop()
            value = exit(path, key, old_parent, new_parent, new_items)
            registry[id_value] = value
            if not new_items_stack:
                continue
        elif id_value in registry:
            value = registry[id_value]
        else:
            res = enter(path, key, value)
            try:
                (new_parent, new_items) = res
            except TypeError:
                raise TypeError('enter should return a tuple of (new_parent, items_iterator), not: %r' % res)
            if new_items is not False:
                registry[id_value] = new_parent
                new_items_stack.append((path, []))
                if value is not root:
                    path += (key,)
                stack.append((_REMAP_EXIT, (key, new_parent, value)))
                if new_items:
                    stack.extend(reversed(list(new_items)))
                continue
        if visit is _orig_default_visit:
            visited_item = (key, value)
        else:
            try:
                visited_item = visit(path, key, value)
            except Exception:
                if reraise_visit:
                    raise
                visited_item = True
            if visited_item is False:
                continue
            elif visited_item is True:
                visited_item = (key, value)
        try:
            new_items_stack[-1][1].append(visited_item)
        except IndexError:
            raise TypeError('expected remappable root, not: %r' % root)
    return value

class PathAccessError(KeyError, IndexError, TypeError):
    """An amalgamation of KeyError, IndexError, and TypeError,
    representing what can occur when looking up a path in a nested
    object.
    """

    def __init__(self, exc, seg, path):
        if False:
            i = 10
            return i + 15
        self.exc = exc
        self.seg = seg
        self.path = path

    def __repr__(self):
        if False:
            return 10
        cn = self.__class__.__name__
        return '%s(%r, %r, %r)' % (cn, self.exc, self.seg, self.path)

    def __str__(self):
        if False:
            print('Hello World!')
        return 'could not access %r from path %r, got error: %r' % (self.seg, self.path, self.exc)

def get_path(root, path, default=_UNSET):
    if False:
        while True:
            i = 10
    "Retrieve a value from a nested object via a tuple representing the\n    lookup path.\n\n    >>> root = {'a': {'b': {'c': [[1], [2], [3]]}}}\n    >>> get_path(root, ('a', 'b', 'c', 2, 0))\n    3\n\n    The path format is intentionally consistent with that of\n    :func:`remap`.\n\n    One of get_path's chief aims is improved error messaging. EAFP is\n    great, but the error messages are not.\n\n    For instance, ``root['a']['b']['c'][2][1]`` gives back\n    ``IndexError: list index out of range``\n\n    What went out of range where? get_path currently raises\n    ``PathAccessError: could not access 2 from path ('a', 'b', 'c', 2,\n    1), got error: IndexError('list index out of range',)``, a\n    subclass of IndexError and KeyError.\n\n    You can also pass a default that covers the entire operation,\n    should the lookup fail at any level.\n\n    Args:\n       root: The target nesting of dictionaries, lists, or other\n          objects supporting ``__getitem__``.\n       path (tuple): A list of strings and integers to be successively\n          looked up within *root*.\n       default: The value to be returned should any\n          ``PathAccessError`` exceptions be raised.\n    "
    if isinstance(path, basestring):
        path = path.split('.')
    cur = root
    try:
        for seg in path:
            try:
                cur = cur[seg]
            except (KeyError, IndexError) as exc:
                raise PathAccessError(exc, seg, path)
            except TypeError as exc:
                try:
                    seg = int(seg)
                    cur = cur[seg]
                except (ValueError, KeyError, IndexError, TypeError):
                    if not is_iterable(cur):
                        exc = TypeError('%r object is not indexable' % type(cur).__name__)
                    raise PathAccessError(exc, seg, path)
    except PathAccessError:
        if default is _UNSET:
            raise
        return default
    return cur

def research(root, query=lambda p, k, v: True, reraise=False):
    if False:
        return 10
    "The :func:`research` function uses :func:`remap` to recurse over\n    any data nested in *root*, and find values which match a given\n    criterion, specified by the *query* callable.\n\n    Results are returned as a list of ``(path, value)`` pairs. The\n    paths are tuples in the same format accepted by\n    :func:`get_path`. This can be useful for comparing values nested\n    in two or more different structures.\n\n    Here's a simple example that finds all integers:\n\n    >>> root = {'a': {'b': 1, 'c': (2, 'd', 3)}, 'e': None}\n    >>> res = research(root, query=lambda p, k, v: isinstance(v, int))\n    >>> print(sorted(res))\n    [(('a', 'b'), 1), (('a', 'c', 0), 2), (('a', 'c', 2), 3)]\n\n    Note how *query* follows the same, familiar ``path, key, value``\n    signature as the ``visit`` and ``enter`` functions on\n    :func:`remap`, and returns a :class:`bool`.\n\n    Args:\n       root: The target object to search. Supports the same types of\n          objects as :func:`remap`, including :class:`list`,\n          :class:`tuple`, :class:`dict`, and :class:`set`.\n       query (callable): The function called on every object to\n          determine whether to include it in the search results. The\n          callable must accept three arguments, *path*, *key*, and\n          *value*, commonly abbreviated *p*, *k*, and *v*, same as\n          *enter* and *visit* from :func:`remap`.\n       reraise (bool): Whether to reraise exceptions raised by *query*\n          or to simply drop the result that caused the error.\n\n\n    With :func:`research` it's easy to inspect the details of a data\n    structure, like finding values that are at a certain depth (using\n    ``len(p)``) and much more. If more advanced functionality is\n    needed, check out the code and make your own :func:`remap`\n    wrapper, and consider `submitting a patch`_!\n\n    .. _submitting a patch: https://github.com/mahmoud/boltons/pulls\n    "
    ret = []
    if not callable(query):
        raise TypeError('query expected callable, not: %r' % query)

    def enter(path, key, value):
        if False:
            i = 10
            return i + 15
        try:
            if query(path, key, value):
                ret.append((path + (key,), value))
        except Exception:
            if reraise:
                raise
        return default_enter(path, key, value)
    remap(root, enter=enter)
    return ret

class GUIDerator(object):
    """The GUIDerator is an iterator that yields a globally-unique
    identifier (GUID) on every iteration. The GUIDs produced are
    hexadecimal strings.

    Testing shows it to be around 12x faster than the uuid module. By
    default it is also more compact, partly due to its default 96-bit
    (24-hexdigit) length. 96 bits of randomness means that there is a
    1 in 2 ^ 32 chance of collision after 2 ^ 64 iterations. If more
    or less uniqueness is desired, the *size* argument can be adjusted
    accordingly.

    Args:
        size (int): character length of the GUID, defaults to 24. Lengths
                    between 20 and 36 are considered valid.

    The GUIDerator has built-in fork protection that causes it to
    detect a fork on next iteration and reseed accordingly.

    """

    def __init__(self, size=24):
        if False:
            while True:
                i = 10
        self.size = size
        if size < 20 or size > 36:
            raise ValueError('expected 20 < size <= 36')
        import hashlib
        self._sha1 = hashlib.sha1
        self.count = itertools.count()
        self.reseed()

    def reseed(self):
        if False:
            print('Hello World!')
        import socket
        self.pid = os.getpid()
        self.salt = '-'.join([str(self.pid), socket.gethostname() or b'<nohostname>', str(time.time()), codecs.encode(os.urandom(6), 'hex_codec').decode('ascii')])
        return

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return self
    if _IS_PY3:

        def __next__(self):
            if False:
                return 10
            if os.getpid() != self.pid:
                self.reseed()
            target_bytes = (self.salt + str(next(self.count))).encode('utf8')
            hash_text = self._sha1(target_bytes).hexdigest()[:self.size]
            return hash_text
    else:

        def __next__(self):
            if False:
                while True:
                    i = 10
            if os.getpid() != self.pid:
                self.reseed()
            return self._sha1(self.salt + str(next(self.count))).hexdigest()[:self.size]
    next = __next__

class SequentialGUIDerator(GUIDerator):
    """Much like the standard GUIDerator, the SequentialGUIDerator is an
    iterator that yields a globally-unique identifier (GUID) on every
    iteration. The GUIDs produced are hexadecimal strings.

    The SequentialGUIDerator differs in that it picks a starting GUID
    value and increments every iteration. This yields GUIDs which are
    of course unique, but also ordered and lexicographically sortable.

    The SequentialGUIDerator is around 50% faster than the normal
    GUIDerator, making it almost 20x as fast as the built-in uuid
    module. By default it is also more compact, partly due to its
    96-bit (24-hexdigit) default length. 96 bits of randomness means that
    there is a 1 in 2 ^ 32 chance of collision after 2 ^ 64
    iterations. If more or less uniqueness is desired, the *size*
    argument can be adjusted accordingly.

    Args:
        size (int): character length of the GUID, defaults to 24.

    Note that with SequentialGUIDerator there is a chance of GUIDs
    growing larger than the size configured. The SequentialGUIDerator
    has built-in fork protection that causes it to detect a fork on
    next iteration and reseed accordingly.

    """
    if _IS_PY3:

        def reseed(self):
            if False:
                while True:
                    i = 10
            super(SequentialGUIDerator, self).reseed()
            start_str = self._sha1(self.salt.encode('utf8')).hexdigest()
            self.start = int(start_str[:self.size], 16)
            self.start |= 1 << self.size * 4 - 2
    else:

        def reseed(self):
            if False:
                while True:
                    i = 10
            super(SequentialGUIDerator, self).reseed()
            start_str = self._sha1(self.salt).hexdigest()
            self.start = int(start_str[:self.size], 16)
            self.start |= 1 << self.size * 4 - 2

    def __next__(self):
        if False:
            print('Hello World!')
        if os.getpid() != self.pid:
            self.reseed()
        return '%x' % (next(self.count) + self.start)
    next = __next__
guid_iter = GUIDerator()
seq_guid_iter = SequentialGUIDerator()

def soft_sorted(iterable, first=None, last=None, key=None, reverse=False):
    if False:
        return 10
    "For when you care about the order of some elements, but not about\n    others.\n\n    Use this to float to the top and/or sink to the bottom a specific\n    ordering, while sorting the rest of the elements according to\n    normal :func:`sorted` rules.\n\n    >>> soft_sorted(['two', 'b', 'one', 'a'], first=['one', 'two'])\n    ['one', 'two', 'a', 'b']\n    >>> soft_sorted(range(7), first=[6, 15], last=[2, 4], reverse=True)\n    [6, 5, 3, 1, 0, 2, 4]\n    >>> import string\n    >>> ''.join(soft_sorted(string.hexdigits, first='za1', last='b', key=str.lower))\n    'aA1023456789cCdDeEfFbB'\n\n    Args:\n       iterable (list): A list or other iterable to sort.\n       first (list): A sequence to enforce for elements which should\n          appear at the beginning of the returned list.\n       last (list): A sequence to enforce for elements which should\n          appear at the end of the returned list.\n       key (callable): Callable used to generate a comparable key for\n          each item to be sorted, same as the key in\n          :func:`sorted`. Note that entries in *first* and *last*\n          should be the keys for the items. Defaults to\n          passthrough/the identity function.\n       reverse (bool): Whether or not elements not explicitly ordered\n          by *first* and *last* should be in reverse order or not.\n\n    Returns a new list in sorted order.\n    "
    first = first or []
    last = last or []
    key = key or (lambda x: x)
    seq = list(iterable)
    other = [x for x in seq if not (first and key(x) in first or (last and key(x) in last))]
    other.sort(key=key, reverse=reverse)
    if first:
        first = sorted([x for x in seq if key(x) in first], key=lambda x: first.index(key(x)))
    if last:
        last = sorted([x for x in seq if key(x) in last], key=lambda x: last.index(key(x)))
    return first + other + last

def untyped_sorted(iterable, key=None, reverse=False):
    if False:
        return 10
    "A version of :func:`sorted` which will happily sort an iterable of\n    heterogeneous types and return a new list, similar to legacy Python's\n    behavior.\n\n    >>> untyped_sorted(['abc', 2.0, 1, 2, 'def'])\n    [1, 2.0, 2, 'abc', 'def']\n\n    Note how mutually orderable types are sorted as expected, as in\n    the case of the integers and floats above.\n\n    .. note::\n\n       Results may vary across Python versions and builds, but the\n       function will produce a sorted list, except in the case of\n       explicitly unorderable objects.\n\n    "

    class _Wrapper(object):
        slots = ('obj',)

        def __init__(self, obj):
            if False:
                return 10
            self.obj = obj

        def __lt__(self, other):
            if False:
                return 10
            obj = key(self.obj) if key is not None else self.obj
            other = key(other.obj) if key is not None else other.obj
            try:
                ret = obj < other
            except TypeError:
                ret = (type(obj).__name__, id(type(obj)), obj) < (type(other).__name__, id(type(other)), other)
            return ret
    if key is not None and (not callable(key)):
        raise TypeError('expected function or callable object for key, not: %r' % key)
    return sorted(iterable, key=_Wrapper, reverse=reverse)
'\nMay actually be faster to do an isinstance check for a str path\n\n$ python -m timeit -s "x = [1]" "x[0]"\n10000000 loops, best of 3: 0.0207 usec per loop\n$ python -m timeit -s "x = [1]" "try: x[0] \nexcept: pass"\n10000000 loops, best of 3: 0.029 usec per loop\n$ python -m timeit -s "x = [1]" "try: x[1] \nexcept: pass"\n1000000 loops, best of 3: 0.315 usec per loop\n# setting up try/except is fast, only around 0.01us\n# actually triggering the exception takes almost 10x as long\n\n$ python -m timeit -s "x = [1]" "isinstance(x, basestring)"\n10000000 loops, best of 3: 0.141 usec per loop\n$ python -m timeit -s "x = [1]" "isinstance(x, str)"\n10000000 loops, best of 3: 0.131 usec per loop\n$ python -m timeit -s "x = [1]" "try: x.split(\'.\')\n except: pass"\n1000000 loops, best of 3: 0.443 usec per loop\n$ python -m timeit -s "x = [1]" "try: x.split(\'.\') \nexcept AttributeError: pass"\n1000000 loops, best of 3: 0.544 usec per loop\n'