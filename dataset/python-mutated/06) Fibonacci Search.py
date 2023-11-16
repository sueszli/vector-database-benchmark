"""
Detailed Explanation at : https://www.geeksforgeeks.org/fibonacci-search/

Runtime: O(log(n))
"""

def fibonacci_search(array, target):
    if False:
        for i in range(10):
            print('nop')
    fibm2 = 0
    fibm1 = 1
    fib = fibm1 + fibm2
    while fib < len(array):
        fibm2 = fibm1
        fibm1 = fib
        fib = fibm1 + fibm2
    offset = -1
    while fib > 1:
        i = min(offset + fibm2, len(array) - 1)
        if array[i] < target:
            fib = fibm1
            fibm1 = fibm2
            fibm2 = fib - fibm1
            offset = i
        elif array[i] > target:
            fib = fibm2
            fibm1 = fibm1 - fibm2
            fibm2 = fib - fibm1
        else:
            return i
    if fibm1 and len(array) - 1 == target:
        return len(array) - 1
    return None

def verify(index, target):
    if False:
        while True:
            i = 10
    if index is not None:
        print('Target', target, 'found at index:', index)
    else:
        print('Target', target, 'not in list')
array = [x for x in range(1, 51)]
print('Input array:', array)
verify(fibonacci_search(array, 30), 30)
verify(fibonacci_search(array, 70), 70)