import collections

class Solution(object):

    def beautifulSubarrays(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        cnt = collections.Counter()
        cnt[0] = 1
        result = curr = 0
        for x in nums:
            curr ^= x
            result += cnt[curr]
            cnt[curr] += 1
        return result