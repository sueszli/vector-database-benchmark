"""
K Closest Points

Given an array with points, and another one point. Find the closest K points of the array to the given point.

Input: [(0, 1), (2, 1), (3, 3), (1, 2)], (2, 2), 2
Output: [(2, 1), (1, 2)]

=========================================
Same solution as the nth_smallest.py, in this case we're looking for the K smallest DISTANCES.
Based on the quick sort algorithm (pivoting, divide and conquer).
More precisly in-place quick sort. Recursive solution.
   Time Complexity:     O(N)    , O(N + N/2 + N/4 + N/8 + ... + 1 = 2*N = N)
   Space Complexity:    O(K)    , length of the output array
Completely the same algorithm as the previous one, but without recursion. This solution is cleaner.
This algorithm is called: QucikSelect - The quicksort pivoting logic but for searching kth smallest (not sorting the whole array) - O(n) complexity (n + n/2 + n/4 + ... + 1 = 2n)
https://en.wikipedia.org/wiki/Quickselect
Same solution as kth_smallest.py.
    Time Complexity:    O(N)
    Space Complexity:   O(K)
"""

def find_k_closes_recursive(arr, pt, k):
    if False:
        print('Hello World!')
    n = len(arr)
    if k > n:
        return arr
    if k < 1:
        return []
    kth_closest(arr, k - 1, 0, n - 1, pt)
    return arr[:k]

def kth_closest(arr, k, left, right, pt):
    if False:
        i = 10
        return i + 15
    pivot = pivoting(arr, left, right, pt)
    if pivot > k:
        kth_closest(arr, k, left, pivot - 1, pt)
    elif pivot < k:
        kth_closest(arr, k, pivot + 1, right, pt)

def pivoting(arr, left, right, pt):
    if False:
        print('Hello World!')
    pivot_dist = sqr_dist(pt, arr[right])
    new_pivot = left
    for j in range(left, right):
        if sqr_dist(pt, arr[j]) < pivot_dist:
            swap(arr, new_pivot, j)
            new_pivot += 1
    swap(arr, new_pivot, right)
    return new_pivot

def swap(arr, i, j):
    if False:
        while True:
            i = 10
    (arr[i], arr[j]) = (arr[j], arr[i])

def sqr_dist(a, b):
    if False:
        while True:
            i = 10
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def find_k_closes(arr, pt, k):
    if False:
        print('Hello World!')
    n = len(arr)
    if k > n:
        return arr
    if k < 1:
        return []
    k -= 1
    left = 0
    right = n - 1
    while True:
        pivot = pivoting(arr, left, right, pt)
        if pivot > k:
            right = pivot - 1
        elif pivot < k:
            left = pivot + 1
        else:
            return arr[:k + 1]
    return None
print(find_k_closes_recursive([(0, 1), (3, 3), (1, 2), (2, 1.5), (3, -1), (2, 1), (4, 3), (5, 1), (-1, 2), (2, 2)], (2, 2), 3))
print(find_k_closes([(0, 1), (3, 3), (1, 2), (2, 1.5), (3, -1), (2, 1), (4, 3), (5, 1), (-1, 2), (2, 2)], (2, 2), 3))
print(find_k_closes_recursive([(0, 1), (2, 1), (3, 3), (1, 2)], (2, 2), 2))
print(find_k_closes([(0, 1), (2, 1), (3, 3), (1, 2)], (2, 2), 2))