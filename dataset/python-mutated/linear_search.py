"""
Linear search works in any array.
T(n): O(n)
"""

def linear_search(array, query):
    if False:
        for i in range(10):
            print('nop')
    "\n    Find the index of the given element in the array.\n    There are no restrictions on the order of the elements in the array.\n    If the element couldn't be found, returns -1.\n    "
    for (i, value) in enumerate(array):
        if value == query:
            return i
    return -1