import collections

class Solution(object):

    def findLHS(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        lookup = collections.defaultdict(int)
        result = 0
        for num in nums:
            lookup[num] += 1
            for diff in [-1, 1]:
                if num + diff in lookup:
                    result = max(result, lookup[num] + lookup[num + diff])
        return result