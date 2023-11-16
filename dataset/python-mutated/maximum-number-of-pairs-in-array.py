class Solution(object):

    def numberOfPairs(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        cnt = [0] * (max(nums) + 1)
        pair_cnt = 0
        for x in nums:
            cnt[x] ^= 1
            if not cnt[x]:
                pair_cnt += 1
        return [pair_cnt, len(nums) - 2 * pair_cnt]
import collections

class Solution2(object):

    def numberOfPairs(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        cnt = collections.Counter(nums)
        pair_cnt = sum((x // 2 for x in cnt.itervalues()))
        return [pair_cnt, len(nums) - 2 * pair_cnt]