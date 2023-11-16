import collections

class Solution(object):

    def longestEqualSubarray(self, nums, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        cnt = collections.Counter()
        result = left = 0
        for right in xrange(len(nums)):
            cnt[nums[right]] += 1
            result = max(result, cnt[nums[right]])
            if right - left + 1 > result + k:
                cnt[nums[left]] -= 1
                left += 1
        return result