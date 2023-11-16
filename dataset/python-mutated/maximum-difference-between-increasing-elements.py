class Solution(object):

    def maximumDifference(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        (result, prefix) = (0, float('inf'))
        for x in nums:
            result = max(result, x - prefix)
            prefix = min(prefix, x)
        return result if result else -1