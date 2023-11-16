class Solution(object):

    def maximumSubarraySum(self, nums, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        result = left = total = 0
        lookup = set()
        for right in xrange(len(nums)):
            while nums[right] in lookup or len(lookup) == k:
                lookup.remove(nums[left])
                total -= nums[left]
                left += 1
            lookup.add(nums[right])
            total += nums[right]
            if len(lookup) == k:
                result = max(result, total)
        return result