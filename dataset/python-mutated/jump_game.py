"""
Jump Game

Given an array of non-negative integers, you are initially positioned at the first index of the array.
Each element in the array represents your maximum jump length at that position.
Determine if you are able to reach the last index.

Input: [2, 3, 1, 1, 4]
Output: True

Input: [3, 2, 1, 0, 4]
Output: False

=========================================
Just iterate the array and in each step save the farthest reachable position.
If the current position is smaller than the farthest position, then the end isn't reachable.
    Time Complexity:    O(N)
    Space Complexity:   O(1)
"""

def can_jump(nums):
    if False:
        while True:
            i = 10
    n = len(nums)
    if n == 0:
        return False
    max_jump = 0
    for i in range(n):
        if max_jump < i:
            return False
        this_jump = i + nums[i]
        max_jump = max(max_jump, this_jump)
        if max_jump >= n - 1:
            return True
print(can_jump([2, 3, 1, 1, 4]))
print(can_jump([3, 2, 1, 0, 4]))