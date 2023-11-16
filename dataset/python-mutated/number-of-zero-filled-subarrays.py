class Solution(object):

    def zeroFilledSubarray(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = 0
        prev = -1
        for i in xrange(len(nums)):
            if nums[i]:
                prev = i
                continue
            result += i - prev
        return result