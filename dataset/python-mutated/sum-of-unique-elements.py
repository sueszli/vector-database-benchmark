import collections

class Solution(object):

    def sumOfUnique(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        return sum((x for (x, c) in collections.Counter(nums).iteritems() if c == 1))