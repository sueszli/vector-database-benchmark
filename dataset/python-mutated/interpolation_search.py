"""
Python implementation of the Interpolation Search algorithm.
Given a sorted array in increasing order, interpolation search calculates
the starting point of its search according to the search key.

FORMULA: start_pos = low + [ (x - arr[low])*(high - low) / (arr[high] - arr[low]) ]

Doc: https://en.wikipedia.org/wiki/Interpolation_search

Time Complexity: O(log2(log2 n)) for average cases, O(n) for the worst case.
The algorithm performs best with uniformly distributed arrays.
"""
from typing import List

def interpolation_search(array: List[int], search_key: int) -> int:
    if False:
        print('Hello World!')
    '\n    :param array: The array to be searched.\n    :param search_key: The key to be searched in the array.\n\n    :returns: Index of search_key in array if found, else -1.\n\n    Examples:\n\n    >>> interpolation_search([-25, -12, -1, 10, 12, 15, 20, 41, 55], -1)\n    2\n    >>> interpolation_search([5, 10, 12, 14, 17, 20, 21], 55)\n    -1\n    >>> interpolation_search([5, 10, 12, 14, 17, 20, 21], -5)\n    -1\n\n    '
    high = len(array) - 1
    low = 0
    while low <= high and array[low] <= search_key <= array[high]:
        pos = low + int((search_key - array[low]) * (high - low) / (array[high] - array[low]))
        if array[pos] == search_key:
            return pos
        if array[pos] < search_key:
            low = pos + 1
        else:
            high = pos - 1
    return -1
if __name__ == '__main__':
    import doctest
    doctest.testmod()