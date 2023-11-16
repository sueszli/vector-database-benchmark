import collections

class Solution(object):

    def minimumSeconds(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        lookup = collections.defaultdict(int)
        dist = collections.defaultdict(int)
        for i in xrange(2 * len(nums)):
            x = nums[i % len(nums)]
            dist[x] = max(dist[x], i - lookup[x])
            lookup[x] = i
        return min(dist.itervalues()) // 2