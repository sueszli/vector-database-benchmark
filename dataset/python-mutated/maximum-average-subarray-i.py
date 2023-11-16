class Solution(object):

    def findMaxAverage(self, nums, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: float\n        '
        result = total = sum(nums[:k])
        for i in xrange(k, len(nums)):
            total += nums[i] - nums[i - k]
            result = max(result, total)
        return float(result) / k