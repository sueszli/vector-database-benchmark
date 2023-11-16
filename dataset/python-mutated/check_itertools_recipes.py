"""Type-annotated versions of the recipes from the itertools docs.

These are all meant to be examples of idiomatic itertools usage,
so they should all type-check without error.
"""
from __future__ import annotations
import collections
import math
import operator
import sys
from itertools import chain, combinations, count, cycle, filterfalse, islice, repeat, starmap, tee, zip_longest
from typing import Any, Callable, Hashable, Iterable, Iterator, Sequence, Tuple, Type, TypeVar, Union, overload
from typing_extensions import Literal, TypeAlias, TypeVarTuple, Unpack
_T = TypeVar('_T')
_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')
_HashableT = TypeVar('_HashableT', bound=Hashable)
_Ts = TypeVarTuple('_Ts')

def take(n: int, iterable: Iterable[_T]) -> list[_T]:
    if False:
        print('Hello World!')
    'Return first n items of the iterable as a list'
    return list(islice(iterable, n))

def prepend(value: _T1, iterator: Iterable[_T2]) -> Iterator[_T1 | _T2]:
    if False:
        while True:
            i = 10
    'Prepend a single value in front of an iterator'
    return chain([value], iterator)

def tabulate(function: Callable[[int], _T], start: int=0) -> Iterator[_T]:
    if False:
        while True:
            i = 10
    'Return function(0), function(1), ...'
    return map(function, count(start))

def repeatfunc(func: Callable[[Unpack[_Ts]], _T], times: int | None=None, *args: Unpack[_Ts]) -> Iterator[_T]:
    if False:
        return 10
    'Repeat calls to func with specified arguments.\n\n    Example:  repeatfunc(random.random)\n    '
    if times is None:
        return starmap(func, repeat(args))
    return starmap(func, repeat(args, times))

def flatten(list_of_lists: Iterable[Iterable[_T]]) -> Iterator[_T]:
    if False:
        print('Hello World!')
    'Flatten one level of nesting'
    return chain.from_iterable(list_of_lists)

def ncycles(iterable: Iterable[_T], n: int) -> Iterator[_T]:
    if False:
        while True:
            i = 10
    'Returns the sequence elements n times'
    return chain.from_iterable(repeat(tuple(iterable), n))

def tail(n: int, iterable: Iterable[_T]) -> Iterator[_T]:
    if False:
        print('Hello World!')
    'Return an iterator over the last n items'
    return iter(collections.deque(iterable, maxlen=n))

def consume(iterator: Iterator[object], n: int | None=None) -> None:
    if False:
        print('Hello World!')
    'Advance the iterator n-steps ahead. If n is None, consume entirely.'
    if n is None:
        collections.deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)

@overload
def nth(iterable: Iterable[_T], n: int, default: None=None) -> _T | None:
    if False:
        print('Hello World!')
    ...

@overload
def nth(iterable: Iterable[_T], n: int, default: _T1) -> _T | _T1:
    if False:
        for i in range(10):
            print('nop')
    ...

def nth(iterable: Iterable[object], n: int, default: object=None) -> object:
    if False:
        return 10
    'Returns the nth item or a default value'
    return next(islice(iterable, n, None), default)

@overload
def quantify(iterable: Iterable[object]) -> int:
    if False:
        i = 10
        return i + 15
    ...

@overload
def quantify(iterable: Iterable[_T], pred: Callable[[_T], bool]) -> int:
    if False:
        while True:
            i = 10
    ...

def quantify(iterable: Iterable[object], pred: Callable[[Any], bool]=bool) -> int:
    if False:
        i = 10
        return i + 15
    'Given a predicate that returns True or False, count the True results.'
    return sum(map(pred, iterable))

@overload
def first_true(iterable: Iterable[_T], default: Literal[False]=False, pred: Callable[[_T], bool] | None=None) -> _T | Literal[False]:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def first_true(iterable: Iterable[_T], default: _T1, pred: Callable[[_T], bool] | None=None) -> _T | _T1:
    if False:
        while True:
            i = 10
    ...

def first_true(iterable: Iterable[object], default: object=False, pred: Callable[[Any], bool] | None=None) -> object:
    if False:
        i = 10
        return i + 15
    'Returns the first true value in the iterable.\n    If no true value is found, returns *default*\n    If *pred* is not None, returns the first item\n    for which pred(item) is true.\n    '
    return next(filter(pred, iterable), default)
_ExceptionOrExceptionTuple: TypeAlias = Union[Type[BaseException], Tuple[Type[BaseException], ...]]

@overload
def iter_except(func: Callable[[], _T], exception: _ExceptionOrExceptionTuple, first: None=None) -> Iterator[_T]:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def iter_except(func: Callable[[], _T], exception: _ExceptionOrExceptionTuple, first: Callable[[], _T1]) -> Iterator[_T | _T1]:
    if False:
        print('Hello World!')
    ...

def iter_except(func: Callable[[], object], exception: _ExceptionOrExceptionTuple, first: Callable[[], object] | None=None) -> Iterator[object]:
    if False:
        while True:
            i = 10
    'Call a function repeatedly until an exception is raised.\n    Converts a call-until-exception interface to an iterator interface.\n    Like builtins.iter(func, sentinel) but uses an exception instead\n    of a sentinel to end the loop.\n    Examples:\n        iter_except(functools.partial(heappop, h), IndexError)   # priority queue iterator\n        iter_except(d.popitem, KeyError)                         # non-blocking dict iterator\n        iter_except(d.popleft, IndexError)                       # non-blocking deque iterator\n        iter_except(q.get_nowait, Queue.Empty)                   # loop over a producer Queue\n        iter_except(s.pop, KeyError)                             # non-blocking set iterator\n    '
    try:
        if first is not None:
            yield first()
        while True:
            yield func()
    except exception:
        pass

def sliding_window(iterable: Iterable[_T], n: int) -> Iterator[tuple[_T, ...]]:
    if False:
        for i in range(10):
            print('nop')
    it = iter(iterable)
    window = collections.deque(islice(it, n - 1), maxlen=n)
    for x in it:
        window.append(x)
        yield tuple(window)

def roundrobin(*iterables: Iterable[_T]) -> Iterator[_T]:
    if False:
        print('Hello World!')
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    num_active = len(iterables)
    nexts: Iterator[Callable[[], _T]] = cycle((iter(it).__next__ for it in iterables))
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))

def partition(pred: Callable[[_T], bool], iterable: Iterable[_T]) -> tuple[Iterator[_T], Iterator[_T]]:
    if False:
        return 10
    'Partition entries into false entries and true entries.\n    If *pred* is slow, consider wrapping it with functools.lru_cache().\n    '
    (t1, t2) = tee(iterable)
    return (filterfalse(pred, t1), filter(pred, t2))

def subslices(seq: Sequence[_T]) -> Iterator[Sequence[_T]]:
    if False:
        print('Hello World!')
    'Return all contiguous non-empty subslices of a sequence'
    slices = starmap(slice, combinations(range(len(seq) + 1), 2))
    return map(operator.getitem, repeat(seq), slices)

def before_and_after(predicate: Callable[[_T], bool], it: Iterable[_T]) -> tuple[Iterator[_T], Iterator[_T]]:
    if False:
        for i in range(10):
            print('nop')
    "Variant of takewhile() that allows complete\n    access to the remainder of the iterator.\n    >>> it = iter('ABCdEfGhI')\n    >>> all_upper, remainder = before_and_after(str.isupper, it)\n    >>> ''.join(all_upper)\n    'ABC'\n    >>> ''.join(remainder)     # takewhile() would lose the 'd'\n    'dEfGhI'\n    Note that the first iterator must be fully\n    consumed before the second iterator can\n    generate valid results.\n    "
    it = iter(it)
    transition: list[_T] = []

    def true_iterator() -> Iterator[_T]:
        if False:
            print('Hello World!')
        for elem in it:
            if predicate(elem):
                yield elem
            else:
                transition.append(elem)
                return

    def remainder_iterator() -> Iterator[_T]:
        if False:
            while True:
                i = 10
        yield from transition
        yield from it
    return (true_iterator(), remainder_iterator())

@overload
def unique_everseen(iterable: Iterable[_HashableT], key: None=None) -> Iterator[_HashableT]:
    if False:
        while True:
            i = 10
    ...

@overload
def unique_everseen(iterable: Iterable[_T], key: Callable[[_T], Hashable]) -> Iterator[_T]:
    if False:
        while True:
            i = 10
    ...

def unique_everseen(iterable: Iterable[_T], key: Callable[[_T], Hashable] | None=None) -> Iterator[_T]:
    if False:
        while True:
            i = 10
    'List unique elements, preserving order. Remember all elements ever seen.'
    seen: set[Hashable] = set()
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen.add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen.add(k)
                yield element

def powerset(iterable: Iterable[_T]) -> Iterator[tuple[_T, ...]]:
    if False:
        for i in range(10):
            print('nop')
    'powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)'
    s = list(iterable)
    return chain.from_iterable((combinations(s, r) for r in range(len(s) + 1)))

def polynomial_derivative(coefficients: Sequence[float]) -> list[float]:
    if False:
        while True:
            i = 10
    "Compute the first derivative of a polynomial.\n    f(x)  =  x³ -4x² -17x + 60\n    f'(x) = 3x² -8x  -17\n    "
    n = len(coefficients)
    powers = reversed(range(1, n))
    return list(map(operator.mul, coefficients, powers))
if sys.version_info >= (3, 8):

    def nth_combination(iterable: Iterable[_T], r: int, index: int) -> tuple[_T, ...]:
        if False:
            print('Hello World!')
        'Equivalent to list(combinations(iterable, r))[index]'
        pool = tuple(iterable)
        n = len(pool)
        c = math.comb(n, r)
        if index < 0:
            index += c
        if index < 0 or index >= c:
            raise IndexError
        result: list[_T] = []
        while r:
            (c, n, r) = (c * r // n, n - 1, r - 1)
            while index >= c:
                index -= c
                (c, n) = (c * (n - r) // n, n - 1)
            result.append(pool[-1 - n])
        return tuple(result)
if sys.version_info >= (3, 10):

    @overload
    def grouper(iterable: Iterable[_T], n: int, *, incomplete: Literal['fill']='fill', fillvalue: None=None) -> Iterator[tuple[_T | None, ...]]:
        if False:
            return 10
        ...

    @overload
    def grouper(iterable: Iterable[_T], n: int, *, incomplete: Literal['fill']='fill', fillvalue: _T1) -> Iterator[tuple[_T | _T1, ...]]:
        if False:
            print('Hello World!')
        ...

    @overload
    def grouper(iterable: Iterable[_T], n: int, *, incomplete: Literal['strict', 'ignore'], fillvalue: None=None) -> Iterator[tuple[_T, ...]]:
        if False:
            i = 10
            return i + 15
        ...

    def grouper(iterable: Iterable[object], n: int, *, incomplete: Literal['fill', 'strict', 'ignore']='fill', fillvalue: object=None) -> Iterator[tuple[object, ...]]:
        if False:
            print('Hello World!')
        'Collect data into non-overlapping fixed-length chunks or blocks'
        args = [iter(iterable)] * n
        if incomplete == 'fill':
            return zip_longest(*args, fillvalue=fillvalue)
        if incomplete == 'strict':
            return zip(*args, strict=True)
        if incomplete == 'ignore':
            return zip(*args)
        else:
            raise ValueError('Expected fill, strict, or ignore')

    def transpose(it: Iterable[Iterable[_T]]) -> Iterator[tuple[_T, ...]]:
        if False:
            i = 10
            return i + 15
        'Swap the rows and columns of the input.'
        return zip(*it, strict=True)
if sys.version_info >= (3, 12):

    def sum_of_squares(it: Iterable[float]) -> float:
        if False:
            for i in range(10):
                print('nop')
        'Add up the squares of the input values.'
        return math.sumprod(*tee(it))

    def convolve(signal: Iterable[float], kernel: Iterable[float]) -> Iterator[float]:
        if False:
            for i in range(10):
                print('nop')
        'Discrete linear convolution of two iterables.\n        The kernel is fully consumed before the calculations begin.\n        The signal is consumed lazily and can be infinite.\n        Convolutions are mathematically commutative.\n        If the signal and kernel are swapped,\n        the output will be the same.\n        Article:  https://betterexplained.com/articles/intuitive-convolution/\n        Video:    https://www.youtube.com/watch?v=KuXjwB4LzSA\n        '
        kernel = tuple(kernel)[::-1]
        n = len(kernel)
        padded_signal = chain(repeat(0, n - 1), signal, repeat(0, n - 1))
        windowed_signal = sliding_window(padded_signal, n)
        return map(math.sumprod, repeat(kernel), windowed_signal)

    def polynomial_eval(coefficients: Sequence[float], x: float) -> float:
        if False:
            while True:
                i = 10
        "Evaluate a polynomial at a specific value.\n        Computes with better numeric stability than Horner's method.\n        "
        n = len(coefficients)
        if not n:
            return type(x)(0)
        powers = map(pow, repeat(x), reversed(range(n)))
        return math.sumprod(coefficients, powers)