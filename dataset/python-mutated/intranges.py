"""
Given a list of integers, made up of (hopefully) a small number of long runs
of consecutive integers, compute a representation of the form
((start1, end1), (start2, end2) ...). Then answer the question "was x present
in the original list?" in time O(log(# runs)).
"""
import bisect
from typing import List, Tuple

def intranges_from_list(list_: List[int]) -> Tuple[int, ...]:
    if False:
        for i in range(10):
            print('nop')
    'Represent a list of integers as a sequence of ranges:\n    ((start_0, end_0), (start_1, end_1), ...), such that the original\n    integers are exactly those x such that start_i <= x < end_i for some i.\n\n    Ranges are encoded as single integers (start << 32 | end), not as tuples.\n    '
    sorted_list = sorted(list_)
    ranges = []
    last_write = -1
    for i in range(len(sorted_list)):
        if i + 1 < len(sorted_list):
            if sorted_list[i] == sorted_list[i + 1] - 1:
                continue
        current_range = sorted_list[last_write + 1:i + 1]
        ranges.append(_encode_range(current_range[0], current_range[-1] + 1))
        last_write = i
    return tuple(ranges)

def _encode_range(start: int, end: int) -> int:
    if False:
        i = 10
        return i + 15
    return start << 32 | end

def _decode_range(r: int) -> Tuple[int, int]:
    if False:
        for i in range(10):
            print('nop')
    return (r >> 32, r & (1 << 32) - 1)

def intranges_contain(int_: int, ranges: Tuple[int, ...]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Determine if `int_` falls into one of the ranges in `ranges`.'
    tuple_ = _encode_range(int_, 0)
    pos = bisect.bisect_left(ranges, tuple_)
    if pos > 0:
        (left, right) = _decode_range(ranges[pos - 1])
        if left <= int_ < right:
            return True
    if pos < len(ranges):
        (left, _) = _decode_range(ranges[pos])
        if left == int_:
            return True
    return False