class Solution(object):

    def maxFrequency(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        left = 0
        nums.sort()
        for right in xrange(len(nums)):
            k += nums[right]
            if k < nums[right] * (right - left + 1):
                k -= nums[left]
                left += 1
        return right - left + 1