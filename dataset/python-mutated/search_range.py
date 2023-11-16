"""
Given an array of integers nums sorted in ascending order, find the starting
and ending position of a given target value. If the target is not found in the
array, return [-1, -1].

For example:
Input: nums = [5,7,7,8,8,8,10], target = 8
Output: [3,5]
Input: nums = [5,7,7,8,8,8,10], target = 11
Output: [-1,-1]
"""

def search_range(nums, target):
    if False:
        print('Hello World!')
    '\n    :type nums: List[int]\n    :type target: int\n    :rtype: List[int]\n    '
    low = 0
    high = len(nums) - 1
    while low < high:
        mid = low + (high - low) // 2
        if target <= nums[mid]:
            high = mid
        else:
            low = mid + 1
    for j in range(len(nums) - 1, -1, -1):
        if nums[j] == target:
            return [low, j]
    return [-1, -1]