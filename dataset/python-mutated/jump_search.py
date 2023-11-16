"""
Jump Search

Find an element in a sorted array.
"""
import math

def jump_search(arr, target):
    if False:
        i = 10
        return i + 15
    '\n    Worst-case Complexity: O(√n) (root(n))\n    All items in list must be sorted like binary search\n\n    Find block that contains target value and search it linearly in that block\n    It returns a first target value in array\n\n    reference: https://en.wikipedia.org/wiki/Jump_search\n    '
    length = len(arr)
    block_size = int(math.sqrt(length))
    block_prev = 0
    block = block_size
    if arr[length - 1] < target:
        return -1
    while block <= length and arr[block - 1] < target:
        block_prev = block
        block += block_size
    while arr[block_prev] < target:
        block_prev += 1
        if block_prev == min(block, length):
            return -1
    if arr[block_prev] == target:
        return block_prev
    return -1