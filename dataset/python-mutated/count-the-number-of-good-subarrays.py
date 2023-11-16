import collections

class Solution(object):

    def countGood(self, nums, k):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        result = curr = left = 0
        cnt = collections.Counter()
        for right in xrange(len(nums)):
            curr += cnt[nums[right]]
            cnt[nums[right]] += 1
            while curr >= k:
                cnt[nums[left]] -= 1
                curr -= cnt[nums[left]]
                left += 1
            result += left
        return result