"""
Representation and utils for ranges of PDF file pages.

Copyright (c) 2014, Steve Witham <switham_github@mac-guyver.com>.
All rights reserved. This software is available under a BSD license;
see https://github.com/py-pdf/pypdf/blob/main/LICENSE
"""
import re
from typing import Any, List, Tuple, Union
from .errors import ParseError
_INT_RE = '(0|-?[1-9]\\d*)'
PAGE_RANGE_RE = f'^({_INT_RE}|({_INT_RE}?(:{_INT_RE}?(:{_INT_RE}?)?)))$'

class PageRange:
    """
    A slice-like representation of a range of page indices.

    For example, page numbers, only starting at zero.

    The syntax is like what you would put between brackets [ ].
    The slice is one of the few Python types that can't be subclassed,
    but this class converts to and from slices, and allows similar use.

      -  PageRange(str) parses a string representing a page range.
      -  PageRange(slice) directly "imports" a slice.
      -  to_slice() gives the equivalent slice.
      -  str() and repr() allow printing.
      -  indices(n) is like slice.indices(n).
    """

    def __init__(self, arg: Union[slice, 'PageRange', str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize with either a slice -- giving the equivalent page range,\n        or a PageRange object -- making a copy,\n        or a string like\n            "int", "[int]:[int]" or "[int]:[int]:[int]",\n            where the brackets indicate optional ints.\n        Remember, page indices start with zero.\n        Page range expression examples:\n\n            :     all pages.                   -1    last page.\n            22    just the 23rd page.          :-1   all but the last page.\n            0:3   the first three pages.       -2    second-to-last page.\n            :3    the first three pages.       -2:   last two pages.\n            5:    from the sixth page onward.  -3:-1 third & second to last.\n        The third, "stride" or "step" number is also recognized.\n            ::2       0 2 4 ... to the end.    3:0:-1    3 2 1 but not 0.\n            1:10:2    1 3 5 7 9                2::-1     2 1 0.\n            ::-1      all pages in reverse order.\n        Note the difference between this notation and arguments to slice():\n            slice(3) means the first three pages;\n            PageRange("3") means the range of only the fourth page.\n            However PageRange(slice(3)) means the first three pages.\n        '
        if isinstance(arg, slice):
            self._slice = arg
            return
        if isinstance(arg, PageRange):
            self._slice = arg.to_slice()
            return
        m = isinstance(arg, str) and re.match(PAGE_RANGE_RE, arg)
        if not m:
            raise ParseError(arg)
        elif m.group(2):
            start = int(m.group(2))
            stop = start + 1 if start != -1 else None
            self._slice = slice(start, stop)
        else:
            self._slice = slice(*[int(g) if g else None for g in m.group(4, 6, 8)])

    @staticmethod
    def valid(input: Any) -> bool:
        if False:
            return 10
        '\n        True if input is a valid initializer for a PageRange.\n\n        Args:\n            input: A possible PageRange string or a PageRange object.\n\n        Returns:\n            True, if the ``input`` is a valid PageRange.\n        '
        return isinstance(input, (slice, PageRange)) or (isinstance(input, str) and bool(re.match(PAGE_RANGE_RE, input)))

    def to_slice(self) -> slice:
        if False:
            for i in range(10):
                print('nop')
        'Return the slice equivalent of this page range.'
        return self._slice

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        'A string like "1:2:3".'
        s = self._slice
        indices: Union[Tuple[int, int], Tuple[int, int, int]]
        if s.step is None:
            if s.start is not None and s.stop == s.start + 1:
                return str(s.start)
            indices = (s.start, s.stop)
        else:
            indices = (s.start, s.stop, s.step)
        return ':'.join(('' if i is None else str(i) for i in indices))

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        'A string like "PageRange(\'1:2:3\')".'
        return 'PageRange(' + repr(str(self)) + ')'

    def indices(self, n: int) -> Tuple[int, int, int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Assuming a sequence of length n, calculate the start and stop indices,\n        and the stride length of the PageRange.\n\n        See help(slice.indices).\n\n        Args:\n            n:  the length of the list of pages to choose from.\n\n        Returns:\n            Arguments for range()\n        '
        return self._slice.indices(n)

    def __eq__(self, other: object) -> bool:
        if False:
            print('Hello World!')
        if not isinstance(other, PageRange):
            return False
        return self._slice == other._slice

    def __add__(self, other: 'PageRange') -> 'PageRange':
        if False:
            while True:
                i = 10
        if not isinstance(other, PageRange):
            raise TypeError(f"Can't add PageRange and {type(other)}")
        if self._slice.step is not None or other._slice.step is not None:
            raise ValueError("Can't add PageRange with stride")
        a = (self._slice.start, self._slice.stop)
        b = (other._slice.start, other._slice.stop)
        if a[0] > b[0]:
            (a, b) = (b, a)
        if b[0] > a[1]:
            raise ValueError("Can't add PageRanges with gap")
        return PageRange(slice(a[0], max(a[1], b[1])))
PAGE_RANGE_ALL = PageRange(':')

def parse_filename_page_ranges(args: List[Union[str, PageRange, None]]) -> List[Tuple[str, PageRange]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a list of filenames and page ranges, return a list of (filename, page_range) pairs.\n\n    Args:\n        args: A list where the first element is a filename. The other elements are\n            filenames, page-range expressions, slice objects, or PageRange objects.\n            A filename not followed by a page range indicates all pages of the file.\n\n    Returns:\n        A list of (filename, page_range) pairs.\n    '
    pairs: List[Tuple[str, PageRange]] = []
    pdf_filename = None
    did_page_range = False
    for arg in args + [None]:
        if PageRange.valid(arg):
            if not pdf_filename:
                raise ValueError('The first argument must be a filename, not a page range.')
            pairs.append((pdf_filename, PageRange(arg)))
            did_page_range = True
        else:
            if pdf_filename and (not did_page_range):
                pairs.append((pdf_filename, PAGE_RANGE_ALL))
            pdf_filename = arg
            did_page_range = False
    return pairs
PageRangeSpec = Union[str, PageRange, Tuple[int, int], Tuple[int, int, int], List[int]]