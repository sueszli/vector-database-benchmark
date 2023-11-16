class Solution(object):

    def minimumDifference(self, nums, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        nums.sort()
        return min((nums[i] - nums[i - k + 1] for i in xrange(k - 1, len(nums))))