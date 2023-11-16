import collections

class Solution(object):

    def findSmallestInteger(self, nums, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type value: int\n        :rtype: int\n        '
        cnt = collections.Counter((x % value for x in nums))
        mn = min(((cnt[i], i) for i in xrange(value)))[1]
        return value * cnt[mn] + mn
import collections

class Solution2(object):

    def findSmallestInteger(self, nums, value):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type value: int\n        :rtype: int\n        '
        cnt = collections.Counter((x % value for x in nums))
        for i in xrange(len(nums) + 1):
            if not cnt[i % value]:
                return i
            cnt[i % value] -= 1