import collections

class Solution(object):

    def mostFrequentEven(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        cnt = collections.Counter((x for x in nums if x % 2 == 0))
        return max(cnt.iterkeys(), key=lambda x: (cnt[x], -x)) if cnt else -1