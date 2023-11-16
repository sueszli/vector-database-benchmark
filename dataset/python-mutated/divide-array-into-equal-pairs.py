import collections

class Solution(object):

    def divideArray(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: bool\n        '
        return all((cnt % 2 == 0 for cnt in collections.Counter(nums).itervalues()))