"""
Longest Increasing Subsequence (LIS)

Find the longest increasing subsequence.
(subsequence doesn't mean that all elements need to be neighboring in the original array).

Sample input: [1, 4, 2, 0, 3, 1]
Sample output: [1, 2, 3]
or output the length
Sample output: 3

=========================================
Dynamic programming (classical) solution.
    Time Complexity:    O(N^2)
    Space Complexity:   O(N)
Dynamic programing in combination with binary search.
Explanation in details: https://www.geeksforgeeks.org/longest-monotonically-increasing-subsequence-size-n-log-n/
    Time Complexity:    O(N * logN)
    Space Complexity:   O(N^2)      , if you need only the length of the LIS, extra space complexity will be O(N)
"""

def longest_increasing_subsequence_1(nums):
    if False:
        i = 10
        return i + 15
    n = len(nums)
    if n == 0:
        return 0
    dp = [1 for i in range(n)]
    max_val = 1
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
                max_val = max(max_val, dp[i])
    current_val = max_val
    result = [0 for i in range(current_val)]
    for i in range(n - 1, -1, -1):
        if dp[i] == current_val and (len(result) == current_val or result[current_val] > nums[i]):
            current_val -= 1
            result[current_val] = nums[i]
    return result

def longest_increasing_subsequence_2(nums):
    if False:
        return 10
    n = len(nums)
    if n == 0:
        return 0
    dp = []
    for i in range(n):
        idx = binary_search(dp, nums[i])
        k = len(dp)
        if idx == k:
            arr = []
            if k != 0:
                arr = [i for i in dp[-1]]
            arr.append(nums[i])
            dp.append(arr)
        elif dp[idx][-1] > nums[i]:
            dp[idx][-1] = nums[i]
    return dp[-1]

def binary_search(dp, target):
    if False:
        for i in range(10):
            print('nop')
    l = 0
    r = len(dp) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if dp[mid][-1] == target:
            return mid
        elif dp[mid][-1] < target:
            l = mid + 1
        else:
            r = mid - 1
    return l
arr = [10, 9, 2, 5, 3, 7, 101, 18]
print(longest_increasing_subsequence_1(arr))
print(longest_increasing_subsequence_2(arr))
arr = [1, 2, 3]
print(longest_increasing_subsequence_1(arr))
print(longest_increasing_subsequence_2(arr))
arr = [10, 1, 3, 8, 2, 0, 5, 7, 12, 3]
print(longest_increasing_subsequence_1(arr))
print(longest_increasing_subsequence_2(arr))
arr = [12, 1, 11, 2, 10, 3, 9, 4, 8, 5, 7, 6]
print(longest_increasing_subsequence_1(arr))
print(longest_increasing_subsequence_2(arr))
arr = [1, 4, 2, 0, 3, 1]
print(longest_increasing_subsequence_1(arr))
print(longest_increasing_subsequence_2(arr))
arr = [7, 5, 5, 5, 5, 5, 3]
print(longest_increasing_subsequence_1(arr))
print(longest_increasing_subsequence_2(arr))