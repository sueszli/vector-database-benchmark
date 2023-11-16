class Solution(object):

    def countSubarrays(self, nums, k):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        result = total = left = 0
        for right in xrange(len(nums)):
            total += nums[right]
            while total * (right - left + 1) >= k:
                total -= nums[left]
                left += 1
            result += right - left + 1
        return result