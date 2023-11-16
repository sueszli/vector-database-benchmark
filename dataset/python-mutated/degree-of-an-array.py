import collections

class Solution(object):

    def findShortestSubArray(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        counts = collections.Counter(nums)
        (left, right) = ({}, {})
        for (i, num) in enumerate(nums):
            left.setdefault(num, i)
            right[num] = i
        degree = max(counts.values())
        return min((right[num] - left[num] + 1 for num in counts.keys() if counts[num] == degree))