import collections

class Solution(object):

    def tupleSameProduct(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = 0
        count = collections.Counter()
        for i in xrange(len(nums)):
            for j in xrange(i + 1, len(nums)):
                result += count[nums[i] * nums[j]]
                count[nums[i] * nums[j]] += 1
        return 8 * result