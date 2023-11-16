from random import randint

class Solution(object):

    def __init__(self, nums):
        if False:
            i = 10
            return i + 15
        '\n\n        :type nums: List[int]\n        :type numsSize: int\n        '
        self.__nums = nums

    def pick(self, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type target: int\n        :rtype: int\n        '
        reservoir = -1
        n = 0
        for i in xrange(len(self.__nums)):
            if self.__nums[i] != target:
                continue
            reservoir = i if randint(1, n + 1) == 1 else reservoir
            n += 1
        return reservoir