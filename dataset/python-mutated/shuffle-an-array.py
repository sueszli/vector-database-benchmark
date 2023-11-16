import random

class Solution(object):

    def __init__(self, nums):
        if False:
            while True:
                i = 10
        '\n\n        :type nums: List[int]\n        :type size: int\n        '
        self.__nums = nums

    def reset(self):
        if False:
            print('Hello World!')
        '\n        Resets the array to its original configuration and return it.\n        :rtype: List[int]\n        '
        return self.__nums

    def shuffle(self):
        if False:
            while True:
                i = 10
        '\n        Returns a random shuffling of the array.\n        :rtype: List[int]\n        '
        nums = list(self.__nums)
        for i in xrange(len(nums)):
            j = random.randint(i, len(nums) - 1)
            (nums[i], nums[j]) = (nums[j], nums[i])
        return nums