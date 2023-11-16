import collections

class Solution(object):

    def countBadPairs(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = len(nums) * (len(nums) - 1) // 2
        cnt = collections.Counter()
        for (i, x) in enumerate(nums):
            result -= cnt[x - i]
            cnt[x - i] += 1
        return result