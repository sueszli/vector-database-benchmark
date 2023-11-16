import collections

class Solution(object):

    def smallerNumbersThanCurrent(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        count = collections.Counter(nums)
        for i in xrange(max(nums) + 1):
            count[i] += count[i - 1]
        return [count[i - 1] for i in nums]
import bisect

class Solution2(object):

    def smallerNumbersThanCurrent(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        sorted_nums = sorted(nums)
        return [bisect.bisect_left(sorted_nums, i) for i in nums]