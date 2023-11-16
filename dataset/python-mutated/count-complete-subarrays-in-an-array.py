import collections

class Solution(object):

    def countCompleteSubarrays(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        nums_set = set(nums)
        result = left = 0
        cnt = collections.Counter()
        for right in xrange(len(nums)):
            cnt[nums[right]] += 1
            while len(cnt) == len(nums_set):
                cnt[nums[left]] -= 1
                if cnt[nums[left]] == 0:
                    del cnt[nums[left]]
                left += 1
            result += left
        return result