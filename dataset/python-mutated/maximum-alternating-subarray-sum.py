class Solution(object):

    def maximumAlternatingSubarraySum(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def kadane(nums, start):
            if False:
                i = 10
                return i + 15
            result = float('-inf')
            curr = odd = 0
            for i in xrange(start, len(nums)):
                curr = curr + nums[i] if not odd else max(curr - nums[i], 0)
                result = max(result, curr)
                odd ^= 1
            return result
        return max(kadane(nums, 0), kadane(nums, 1))