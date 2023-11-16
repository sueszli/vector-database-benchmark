import collections

class Solution(object):

    def countQuadruplets(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = 0
        lookup = collections.defaultdict(int)
        lookup[nums[-1]] = 1
        for c in reversed(xrange(2, len(nums) - 1)):
            for b in xrange(1, c):
                for a in xrange(b):
                    if nums[a] + nums[b] + nums[c] in lookup:
                        result += lookup[nums[a] + nums[b] + nums[c]]
            lookup[nums[c]] += 1
        return result
import collections

class Solution2(object):

    def countQuadruplets(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        lookup = collections.defaultdict(list)
        for d in xrange(3, len(nums)):
            for c in xrange(2, d):
                lookup[nums[d] - nums[c]].append(c)
        return sum((sum((b < c for c in lookup[nums[a] + nums[b]])) for b in xrange(1, len(nums) - 2) for a in xrange(b)))