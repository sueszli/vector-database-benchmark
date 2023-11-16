class Solution(object):

    def maximumTop(self, nums, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        if len(nums) == 1 == k % 2:
            return -1
        if k <= 1:
            return nums[k]
        return max((nums[i] for i in xrange(min(k + 1, len(nums))) if i != k - 1))