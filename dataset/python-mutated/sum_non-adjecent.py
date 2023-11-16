"""
Sum of non-adjacent numbers

Given a list of integers, write a function that returns the largest sum of non-adjacent numbers.
Numbers can be 0 or negative.

Input: [2, 4, 6, 2, 5]
Output: 13
Output explanation: We pick 2, 6, and 5.

Input: [5, 1, 1, 5]
Output: 10
Output explanation: We pick 5 and 5.

=========================================
Dynamic programming solution, but don't need the whole DP array, only the last 3 sums (DPs) are needed.
    Time Complexity:    O(N)
    Space Complexity:   O(1)
"""

def sum_non_adjacent(arr):
    if False:
        for i in range(10):
            print('nop')
    n = len(arr)
    sums = [0, 0, 0]
    if n == 0:
        return 0
    sums[0] = max(arr[0], 0)
    if n == 1:
        return sums[0]
    sums[1] = arr[1]
    if sums[1] <= 0:
        sums[1] = sums[0]
    if n == 2:
        return max(sums[0], sums[1])
    sums[2] = arr[2]
    if sums[2] <= 0:
        sums[2] = max(sums[0], sums[1])
    else:
        sums[2] += sums[0]
    for i in range(3, n):
        temp = 0
        if arr[i] > 0:
            temp = max(sums[0], sums[1]) + arr[i]
        else:
            temp = max(sums)
        sums = sums[1:] + [temp]
    return max(sums)
print(sum_non_adjacent([2, 4, 6, 2, 5]))
print(sum_non_adjacent([2, 4, 2, 6, 2, -3, -2, 0, -3, 5]))
print(sum_non_adjacent([5, 1, 1, 5]))
print(sum_non_adjacent([5, 1, -1, 1, 5]))