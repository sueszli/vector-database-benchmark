"""
K-th Smallest Number

Find the K-th smallest number in an unordered list.

Input: [6, 2, 4, 8, 10, 1, 11], 1
Output: 0

Input: [6, 2, 4, 8, 10, 0, 11], 2
Output: 2

Input: [6, 2, 4, 8, 10, 0, 11], 4
Output: 6

=========================================
This solution is based on the quick sort algorithm (pivoting, divide and conquer).
More precisly in-place quick sort. Recursive solution.
   Time Complexity:     O(N)    , O(N + N/2 + N/4 + N/8 + ... + 1 = 2*N = N)
   Space Complexity:    O(LogN) , because of the recursion stack
Completely the same algorithm as the previous one, but without recursion. This solution is cleaner.
This algorithm is called: QucikSelect - The quicksort pivoting logic but for searching kth smallest (not sorting the whole array) - O(n) complexity (n + n/2 + n/4 + ... + 1 = 2n)
https://en.wikipedia.org/wiki/Quickselect
    Time Complexity:    O(N)
    Space Complexity:   O(1)
"""

def find_kth_smallest_recursive(arr, k):
    if False:
        return 10
    n = len(arr)
    if k > n:
        return None
    if k < 1:
        return None
    return kth_smallest(arr, k - 1, 0, n - 1)

def kth_smallest(arr, k, left, right):
    if False:
        i = 10
        return i + 15
    pivot = pivoting(arr, left, right)
    if pivot > k:
        return kth_smallest(arr, k, left, pivot - 1)
    if pivot < k:
        return kth_smallest(arr, k, pivot + 1, right)
    return arr[pivot]

def pivoting(arr, left, right):
    if False:
        while True:
            i = 10
    pivot = right
    new_pivot = left
    for j in range(left, right):
        if arr[j] < arr[pivot]:
            swap(arr, new_pivot, j)
            new_pivot += 1
    swap(arr, new_pivot, pivot)
    return new_pivot

def swap(arr, i, j):
    if False:
        print('Hello World!')
    (arr[i], arr[j]) = (arr[j], arr[i])

def find_kth_smallest(arr, k):
    if False:
        while True:
            i = 10
    n = len(arr)
    if k > n:
        return None
    if k < 1:
        return None
    k -= 1
    left = 0
    right = n - 1
    while True:
        pivot = pivoting(arr, left, right)
        if pivot > k:
            right = pivot - 1
        elif pivot < k:
            left = pivot + 1
        else:
            return arr[pivot]
    return None
arr = [1, 1, 1, 1, 1, 1]
print([find_kth_smallest_recursive(arr, i) for i in range(1, len(arr) + 1)])
print([find_kth_smallest(arr, i) for i in range(1, len(arr) + 1)])
arr = [6, 4, 2, 12, 4, 8, 10, 1, 11, 0, 8, 4]
print([find_kth_smallest_recursive(arr, i) for i in range(1, len(arr) + 1)])
print([find_kth_smallest(arr, i) for i in range(1, len(arr) + 1)])
arr = [5, 4, 3, 2, 1]
print([find_kth_smallest_recursive(arr, i) for i in range(1, len(arr) + 1)])
print([find_kth_smallest(arr, i) for i in range(1, len(arr) + 1)])