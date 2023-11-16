"""
Given an array of integers, every element appears
twice except for one. Find that single one.

NOTE: This also works for finding a number occurring odd
      number of times, where all the other numbers appear
      even number of times.

Note:
Your algorithm should have a linear runtime complexity.
Could you implement it without using extra memory?
"""

def single_number(nums):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns single number, if found.\n    Else if all numbers appear twice, returns 0.\n    :type nums: List[int]\n    :rtype: int\n    '
    i = 0
    for num in nums:
        i ^= num
    return i