class Solution(object):

    def longestNiceSubarray(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = left = curr = 0
        for right in xrange(len(nums)):
            while curr & nums[right]:
                curr ^= nums[left]
                left += 1
            curr |= nums[right]
            result = max(result, right - left + 1)
        return result