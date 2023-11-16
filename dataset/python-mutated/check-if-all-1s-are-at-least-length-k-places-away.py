class Solution(object):

    def kLengthApart(self, nums, k):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: bool\n        '
        prev = -k - 1
        for i in xrange(len(nums)):
            if not nums[i]:
                continue
            if i - prev <= k:
                return False
            prev = i
        return True