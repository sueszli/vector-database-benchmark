class Solution(object):

    def getAverages(self, nums, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: List[int]\n        '
        (total, l) = (0, 2 * k + 1)
        result = [-1] * len(nums)
        for i in xrange(len(nums)):
            total += nums[i]
            if i - l >= 0:
                total -= nums[i - l]
            if i >= l - 1:
                result[i - k] = total // l
        return result