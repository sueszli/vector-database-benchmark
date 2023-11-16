import collections

class Solution(object):

    def numIdenticalPairs(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        return sum((c * (c - 1) // 2 for c in collections.Counter(nums).itervalues()))