"""
Longest Increasing Subarray

Find the longest increasing subarray (subarray is when all elements are neighboring in the original array).

Input: [10, 1, 3, 8, 2, 0, 5, 7, 12, 3]
Output: 4

=========================================
Only in one iteration, check if the current element is bigger than the previous and increase the counter if true.
    Time Complexity:    O(N)
    Space Complexity:   O(1)
"""

def longest_increasing_subarray(arr):
    if False:
        while True:
            i = 10
    n = len(arr)
    longest = 0
    current = 1
    i = 1
    while i < n:
        if arr[i] < arr[i - 1]:
            longest = max(longest, current)
            current = 1
        else:
            current += 1
        i += 1
    return max(longest, current)
print(longest_increasing_subarray([10, 1, 3, 8, 2, 0, 5, 7, 12, 3]))