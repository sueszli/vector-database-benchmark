"""A module for miscellaneous useful bits and bobs that don't
obviously belong anywhere else. If you spot a better home for
anything that lives here, please move it."""
import array
import sys
import warnings
from random import Random
from typing import Callable, Dict, Generic, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union, overload
from hypothesis.errors import HypothesisWarning
ARRAY_CODES = ['B', 'H', 'I', 'L', 'Q', 'O']

def array_or_list(code: str, contents: Iterable[int]) -> 'Union[List[int], array.ArrayType[int]]':
    if False:
        i = 10
        return i + 15
    if code == 'O':
        return list(contents)
    return array.array(code, contents)

def replace_all(buffer: Sequence[int], replacements: Iterable[Tuple[int, int, Sequence[int]]]) -> bytes:
    if False:
        i = 10
        return i + 15
    'Substitute multiple replacement values into a buffer.\n\n    Replacements is a list of (start, end, value) triples.\n    '
    result = bytearray()
    prev = 0
    offset = 0
    for (u, v, r) in replacements:
        result.extend(buffer[prev:u])
        result.extend(r)
        prev = v
        offset += len(r) - (v - u)
    result.extend(buffer[prev:])
    assert len(result) == len(buffer) + offset
    return bytes(result)
NEXT_ARRAY_CODE = dict(zip(ARRAY_CODES, ARRAY_CODES[1:]))

class IntList(Sequence[int]):
    """Class for storing a list of non-negative integers compactly.

    We store them as the smallest size integer array we can get
    away with. When we try to add an integer that is too large,
    we upgrade the array to the smallest word size needed to store
    the new value."""
    __slots__ = ('__underlying',)
    __underlying: 'Union[List[int], array.ArrayType[int]]'

    def __init__(self, values: Sequence[int]=()):
        if False:
            i = 10
            return i + 15
        for code in ARRAY_CODES:
            try:
                underlying = array_or_list(code, values)
                break
            except OverflowError:
                pass
        else:
            raise AssertionError(f'Could not create storage for {values!r}')
        if isinstance(underlying, list):
            for v in underlying:
                if not isinstance(v, int) or v < 0:
                    raise ValueError(f'Could not create IntList for {values!r}')
        self.__underlying = underlying

    @classmethod
    def of_length(cls, n: int) -> 'IntList':
        if False:
            return 10
        return cls(array_or_list('B', [0]) * n)

    def count(self, value: int) -> int:
        if False:
            i = 10
            return i + 15
        return self.__underlying.count(value)

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'IntList({list(self.__underlying)!r})'

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.__underlying)

    @overload
    def __getitem__(self, i: int) -> int:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def __getitem__(self, i: slice) -> 'IntList':
        if False:
            print('Hello World!')
        ...

    def __getitem__(self, i: Union[int, slice]) -> 'Union[int, IntList]':
        if False:
            while True:
                i = 10
        if isinstance(i, slice):
            return IntList(self.__underlying[i])
        return self.__underlying[i]

    def __delitem__(self, i: int) -> None:
        if False:
            while True:
                i = 10
        del self.__underlying[i]

    def insert(self, i: int, v: int) -> None:
        if False:
            return 10
        self.__underlying.insert(i, v)

    def __iter__(self) -> Iterator[int]:
        if False:
            print('Hello World!')
        return iter(self.__underlying)

    def __eq__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        if self is other:
            return True
        if not isinstance(other, IntList):
            return NotImplemented
        return self.__underlying == other.__underlying

    def __ne__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        if self is other:
            return False
        if not isinstance(other, IntList):
            return NotImplemented
        return self.__underlying != other.__underlying

    def append(self, n: int) -> None:
        if False:
            i = 10
            return i + 15
        i = len(self)
        self.__underlying.append(0)
        self[i] = n

    def __setitem__(self, i: int, n: int) -> None:
        if False:
            print('Hello World!')
        while True:
            try:
                self.__underlying[i] = n
                return
            except OverflowError:
                assert n > 0
                self.__upgrade()

    def extend(self, ls: Iterable[int]) -> None:
        if False:
            while True:
                i = 10
        for n in ls:
            self.append(n)

    def __upgrade(self) -> None:
        if False:
            i = 10
            return i + 15
        assert isinstance(self.__underlying, array.array)
        code = NEXT_ARRAY_CODE[self.__underlying.typecode]
        self.__underlying = array_or_list(code, self.__underlying)

def binary_search(lo: int, hi: int, f: Callable[[int], bool]) -> int:
    if False:
        return 10
    'Binary searches in [lo , hi) to find\n    n such that f(n) == f(lo) but f(n + 1) != f(lo).\n    It is implicitly assumed and will not be checked\n    that f(hi) != f(lo).\n    '
    reference = f(lo)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if f(mid) == reference:
            lo = mid
        else:
            hi = mid
    return lo

def uniform(random: Random, n: int) -> bytes:
    if False:
        i = 10
        return i + 15
    'Returns a bytestring of length n, distributed uniformly at random.'
    return random.getrandbits(n * 8).to_bytes(n, 'big')
T = TypeVar('T')

class LazySequenceCopy:
    """A "copy" of a sequence that works by inserting a mask in front
    of the underlying sequence, so that you can mutate it without changing
    the underlying sequence. Effectively behaves as if you could do list(x)
    in O(1) time. The full list API is not supported yet but there's no reason
    in principle it couldn't be."""
    __mask: Optional[Dict[int, int]]

    def __init__(self, values: Sequence[int]):
        if False:
            for i in range(10):
                print('nop')
        self.__values = values
        self.__len = len(values)
        self.__mask = None

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return self.__len

    def pop(self) -> int:
        if False:
            while True:
                i = 10
        if len(self) == 0:
            raise IndexError('Cannot pop from empty list')
        result = self[-1]
        self.__len -= 1
        if self.__mask is not None:
            self.__mask.pop(self.__len, None)
        return result

    def __getitem__(self, i: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        i = self.__check_index(i)
        default = self.__values[i]
        if self.__mask is None:
            return default
        else:
            return self.__mask.get(i, default)

    def __setitem__(self, i: int, v: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        i = self.__check_index(i)
        if self.__mask is None:
            self.__mask = {}
        self.__mask[i] = v

    def __check_index(self, i: int) -> int:
        if False:
            i = 10
            return i + 15
        n = len(self)
        if i < -n or i >= n:
            raise IndexError(f'Index {i} out of range [0, {n})')
        if i < 0:
            i += n
        assert 0 <= i < n
        return i

def clamp(lower: int, value: int, upper: int) -> int:
    if False:
        i = 10
        return i + 15
    "Given a value and lower/upper bounds, 'clamp' the value so that\n    it satisfies lower <= value <= upper."
    return max(lower, min(value, upper))

def swap(ls: LazySequenceCopy, i: int, j: int) -> None:
    if False:
        while True:
            i = 10
    'Swap the elements ls[i], ls[j].'
    if i == j:
        return
    (ls[i], ls[j]) = (ls[j], ls[i])

def stack_depth_of_caller() -> int:
    if False:
        while True:
            i = 10
    "Get stack size for caller's frame.\n\n    From https://stackoverflow.com/a/47956089/9297601 , this is a simple\n    but much faster alternative to `len(inspect.stack(0))`.  We use it\n    with get/set recursionlimit to make stack overflows non-flaky; see\n    https://github.com/HypothesisWorks/hypothesis/issues/2494 for details.\n    "
    frame = sys._getframe(2)
    size = 1
    while frame:
        frame = frame.f_back
        size += 1
    return size

class ensure_free_stackframes:
    """Context manager that ensures there are at least N free stackframes (for
    a reasonable value of N).
    """

    def __enter__(self):
        if False:
            while True:
                i = 10
        cur_depth = stack_depth_of_caller()
        self.old_maxdepth = sys.getrecursionlimit()
        self.new_maxdepth = cur_depth + 2000
        assert cur_depth <= 1000, 'Hypothesis would usually add %d to the stack depth of %d here, but we are already much deeper than expected.  Aborting now, to avoid extending the stack limit in an infinite loop...' % (self.new_maxdepth - self.old_maxdepth, self.old_maxdepth)
        sys.setrecursionlimit(self.new_maxdepth)

    def __exit__(self, *args, **kwargs):
        if False:
            return 10
        if self.new_maxdepth == sys.getrecursionlimit():
            sys.setrecursionlimit(self.old_maxdepth)
        else:
            warnings.warn('The recursion limit will not be reset, since it was changed from another thread or during execution of a test.', HypothesisWarning, stacklevel=2)

def find_integer(f: Callable[[int], bool]) -> int:
    if False:
        print('Hello World!')
    'Finds a (hopefully large) integer such that f(n) is True and f(n + 1) is\n    False.\n\n    f(0) is assumed to be True and will not be checked.\n    '
    for i in range(1, 5):
        if not f(i):
            return i - 1
    lo = 4
    hi = 5
    while f(hi):
        lo = hi
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if f(mid):
            lo = mid
        else:
            hi = mid
    return lo

def pop_random(random: Random, seq: LazySequenceCopy) -> int:
    if False:
        print('Hello World!')
    'Remove and return a random element of seq. This runs in O(1) but leaves\n    the sequence in an arbitrary order.'
    i = random.randrange(0, len(seq))
    swap(seq, i, len(seq) - 1)
    return seq.pop()

class NotFound(Exception):
    pass

class SelfOrganisingList(Generic[T]):
    """A self-organising list with the move-to-front heuristic.

    A self-organising list is a collection which we want to retrieve items
    that satisfy some predicate from. There is no faster way to do this than
    a linear scan (as the predicates may be arbitrary), but the performance
    of a linear scan can vary dramatically - if we happen to find a good item
    on the first try it's O(1) after all. The idea of a self-organising list is
    to reorder the list to try to get lucky this way as often as possible.

    There are various heuristics we could use for this, and it's not clear
    which are best. We use the simplest, which is that every time we find
    an item we move it to the "front" (actually the back in our implementation
    because we iterate in reverse) of the list.

    """

    def __init__(self, values: Iterable[T]=()) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.__values = list(values)

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'SelfOrganisingList({self.__values!r})'

    def add(self, value: T) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Add a value to this list.'
        self.__values.append(value)

    def find(self, condition: Callable[[T], bool]) -> T:
        if False:
            for i in range(10):
                print('nop')
        'Returns some value in this list such that ``condition(value)`` is\n        True. If no such value exists raises ``NotFound``.'
        for i in range(len(self.__values) - 1, -1, -1):
            value = self.__values[i]
            if condition(value):
                del self.__values[i]
                self.__values.append(value)
                return value
        raise NotFound('No values satisfying condition')