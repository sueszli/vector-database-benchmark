class Solution(object):

    def findFinalValue(self, nums, original):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type original: int\n        :rtype: int\n        '
        lookup = set(nums)
        while original in lookup:
            original *= 2
        return original