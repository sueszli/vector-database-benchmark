from sortedcontainers import SortedList

class Solution(object):

    def minAbsoluteDifference(self, nums, x):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type x: int\n        :rtype: int\n        '
        result = float('inf')
        sl = SortedList()
        for i in xrange(x, len(nums)):
            sl.add(nums[i - x])
            j = sl.bisect_left(nums[i])
            if j - 1 >= 0:
                result = min(result, nums[i] - sl[j - 1])
            if j < len(sl):
                result = min(result, sl[j] - nums[i])
        return result