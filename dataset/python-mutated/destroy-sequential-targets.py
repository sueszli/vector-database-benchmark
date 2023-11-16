import collections

class Solution(object):

    def destroyTargets(self, nums, space):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type space: int\n        :rtype: int\n        '
        cnt = collections.Counter((x % space for x in nums))
        mx = max(cnt.itervalues())
        return min((x for x in nums if cnt[x % space] == mx))