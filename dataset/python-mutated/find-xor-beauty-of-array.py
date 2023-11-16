import operator

class Solution(object):

    def xorBeauty(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        return reduce(operator.xor, nums)