import collections

class Solution(object):

    def countInterestingSubarrays(self, nums, modulo, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type modulo: int\n        :type k: int\n        :rtype: int\n        '
        cnt = collections.Counter([0])
        result = prefix = 0
        for x in nums:
            if x % modulo == k:
                prefix = (prefix + 1) % modulo
            result += cnt[(prefix - k) % modulo]
            cnt[prefix] += 1
        return result