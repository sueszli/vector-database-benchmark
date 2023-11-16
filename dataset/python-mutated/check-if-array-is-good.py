class Solution(object):

    def isGood(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: bool\n        '
        cnt = [0] * len(nums)
        for x in nums:
            if x < len(cnt):
                cnt[x] += 1
            else:
                return False
        return all((cnt[x] == 1 for x in xrange(1, len(nums) - 1)))