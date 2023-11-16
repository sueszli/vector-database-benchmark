import collections

class Solution(object):

    def maxSum(self, nums, m, k):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type m: int\n        :type k: int\n        :rtype: int\n        '
        lookup = collections.Counter()
        result = curr = left = 0
        for right in xrange(len(nums)):
            curr += nums[right]
            lookup[nums[right]] += 1
            if right - left + 1 == k + 1:
                lookup[nums[left]] -= 1
                if lookup[nums[left]] == 0:
                    del lookup[nums[left]]
                curr -= nums[left]
                left += 1
            if right - left + 1 == k and len(lookup) >= m:
                result = max(result, curr)
        return result