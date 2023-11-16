"""
Subarray with given sum

Given an unsorted array A of size N of non-negative integers, find a continuous sub-array
which adds to a given number. Find starting and ending positions(1 indexing) of first such
occuring subarray from the left if sum equals to subarray, else print -1.

Input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 15
Output: 1, 5

=========================================
Adjust the start and end index, in each step increase start or end idx.
If sum is bigger than K, remove element from the start idx from the sum.
Else add element from the end idx to the sum.
    Time Complexity:    O(N)
    Space Complexity:   O(1)
"""

def find_subarray(arr, k):
    if False:
        print('Hello World!')
    n = len(arr)
    if n == 0:
        return -1
    start = 0
    end = 0
    current_sum = arr[0]
    while end < n:
        if current_sum == k:
            return (start + 1, end + 1)
        if current_sum < k:
            end += 1
            current_sum += arr[end]
        else:
            current_sum -= arr[start]
            start += 1
    return -1
print(find_subarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 15))
print(find_subarray([1, 2, 3, 7, 5], 12))
print(find_subarray([6, 6, 6, 6, 3], 3))