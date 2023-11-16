"""
Binary Search

Find an element in a sorted array (in ascending order).
"""

def binary_search(array, query):
    if False:
        i = 10
        return i + 15
    '\n    Worst-case Complexity: O(log(n))\n\n    reference: https://en.wikipedia.org/wiki/Binary_search_algorithm\n    '
    (low, high) = (0, len(array) - 1)
    while low <= high:
        mid = (high + low) // 2
        val = array[mid]
        if val == query:
            return mid
        if val < query:
            low = mid + 1
        else:
            high = mid - 1
    return None

def binary_search_recur(array, low, high, val):
    if False:
        while True:
            i = 10
    '\n    Worst-case Complexity: O(log(n))\n\n    reference: https://en.wikipedia.org/wiki/Binary_search_algorithm\n    '
    if low > high:
        return -1
    mid = low + (high - low) // 2
    if val < array[mid]:
        return binary_search_recur(array, low, mid - 1, val)
    if val > array[mid]:
        return binary_search_recur(array, mid + 1, high, val)
    return mid