import sys
import cython
try:
    from builtins import next
except ImportError:

    def next(it):
        if False:
            return 10
        return it.next()

def for_in_pyiter_pass(it):
    if False:
        print('Hello World!')
    '\n    >>> it = Iterable(5)\n    >>> for_in_pyiter_pass(it)\n    >>> next(it)\n    Traceback (most recent call last):\n    StopIteration\n    '
    for item in it:
        pass

def for_in_pyiter(it):
    if False:
        while True:
            i = 10
    '\n    >>> for_in_pyiter(Iterable(5))\n    [0, 1, 2, 3, 4]\n    '
    l = []
    for item in it:
        l.append(item)
    return l

def for_in_list():
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> for_in_pyiter([1,2,3,4,5])\n    [1, 2, 3, 4, 5]\n    '

@cython.test_assert_path_exists('//TupleNode//IntNode')
@cython.test_fail_if_path_exists('//ListNode//IntNode')
def for_in_literal_list():
    if False:
        i = 10
        return i + 15
    '\n    >>> for_in_literal_list()\n    [1, 2, 3, 4]\n    '
    l = []
    for i in [1, 2, 3, 4]:
        l.append(i)
    return l

@cython.test_assert_path_exists('//TupleNode//IntNode')
@cython.test_fail_if_path_exists('//ListNode//IntNode')
def for_in_literal_mult_list():
    if False:
        i = 10
        return i + 15
    '\n    >>> for_in_literal_mult_list()\n    [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]\n    '
    l = []
    for i in [1, 2, 3, 4] * 3:
        l.append(i)
    return l

def listcomp_over_multiplied_constant_tuple():
    if False:
        while True:
            i = 10
    '\n    >>> listcomp_over_multiplied_constant_tuple()\n    [[], [1, 2, 3], [1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]]\n    '
    return [[i for i in (1, 2, 3) * 0], [i for i in (1, 2, 3) * 1], [i for i in (1, 2, 3) * 2], [i for i in (1, 2, 3) * 3], [i for i in (1, 2, 3) * 2]]

@cython.test_assert_path_exists('//ReturnStatNode//ForInStatNode//TupleNode')
@cython.test_fail_if_path_exists('//ReturnStatNode//ForInStatNode//ListNode')
def listcomp_over_multiplied_constant_list():
    if False:
        i = 10
        return i + 15
    '\n    >>> listcomp_over_multiplied_constant_list()\n    [[], [1, 2, 3], [1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]]\n    '
    return [[i for i in [1, 2, 3] * 0], [i for i in [1, 2, 3] * 1], [i for i in [1, 2, 3] * 2], [i for i in [1, 2, 3] * 3], [i for i in [1, 2, 3] * 2]]

class Iterable(object):
    """
    >>> for_in_pyiter(Iterable(5))
    [0, 1, 2, 3, 4]
    """

    def __init__(self, N):
        if False:
            for i in range(10):
                print('nop')
        self.N = N
        self.i = 0

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __next__(self):
        if False:
            print('Hello World!')
        if self.i < self.N:
            i = self.i
            self.i += 1
            return i
        raise StopIteration
    next = __next__

class NextReplacingIterable(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.i = 0

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

    def __next__(self):
        if False:
            while True:
                i = 10
        if self.i > 5:
            raise StopIteration
        self.i += 1
        self.__next__ = self.next2
        return 1

    def next2(self):
        if False:
            for i in range(10):
                print('nop')
        self.__next__ = self.next3
        return 2

    def next3(self):
        if False:
            while True:
                i = 10
        del self.__next__
        raise StopIteration

def for_in_next_replacing_iter():
    if False:
        while True:
            i = 10
    '\n    >>> for_in_pyiter(NextReplacingIterable())\n    [1, 1, 1, 1, 1, 1]\n    '

def for_in_gen(N):
    if False:
        while True:
            i = 10
    '\n    >>> for_in_pyiter(for_in_gen(10))\n    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\n    '
    for i in range(N):
        yield i

def for_in_range_invalid_arg_count():
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> for_in_range_invalid_arg_count()     # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: ...\n    '
    for i in range(1, 2, 3, 4):
        pass