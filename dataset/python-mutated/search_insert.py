"""
Helper methods for implementing insertion sort.
"""

def search_insert(array, val):
    if False:
        while True:
            i = 10
    '\n    Given a sorted array and a target value, return the index if the target is\n    found. If not, return the index where it would be if it were inserted in order.\n\n    For example:\n    [1,3,5,6], 5 -> 2\n    [1,3,5,6], 2 -> 1\n    [1,3,5,6], 7 -> 4\n    [1,3,5,6], 0 -> 0\n    '
    low = 0
    high = len(array) - 1
    while low <= high:
        mid = low + (high - low) // 2
        if val > array[mid]:
            low = mid + 1
        else:
            high = mid - 1
    return low