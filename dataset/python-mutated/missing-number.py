import operator

class Solution(object):

    def missingNumber(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        return reduce(operator.xor, nums, reduce(operator.xor, xrange(len(nums) + 1)))

class Solution2(object):

    def missingNumber(self, nums):
        if False:
            print('Hello World!')
        return sum(xrange(len(nums) + 1)) - sum(nums)