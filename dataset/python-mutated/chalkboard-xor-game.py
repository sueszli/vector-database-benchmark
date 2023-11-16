from operator import xor
from functools import reduce

class Solution(object):

    def xorGame(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: bool\n        '
        return reduce(xor, nums) == 0 or len(nums) % 2 == 0