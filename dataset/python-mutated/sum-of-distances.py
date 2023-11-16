import collections

class Solution(object):

    def distance(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        result = [0] * len(nums)
        (cnt1, left) = (collections.Counter(), collections.Counter())
        for i in xrange(len(nums)):
            result[i] += cnt1[nums[i]] * i - left[nums[i]]
            cnt1[nums[i]] += 1
            left[nums[i]] += i
        (cnt2, right) = (collections.Counter(), collections.Counter())
        for i in reversed(xrange(len(nums))):
            result[i] += right[nums[i]] - cnt2[nums[i]] * i
            cnt2[nums[i]] += 1
            right[nums[i]] += i
        return result