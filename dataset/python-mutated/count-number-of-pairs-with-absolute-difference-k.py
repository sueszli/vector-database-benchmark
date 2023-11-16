import collections

class Solution(object):

    def countKDifference(self, nums, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        lookup = collections.defaultdict(int)
        result = 0
        for x in nums:
            if x - k in lookup:
                result += lookup[x - k]
            if x + k in lookup:
                result += lookup[x + k]
            lookup[x] += 1
        return result