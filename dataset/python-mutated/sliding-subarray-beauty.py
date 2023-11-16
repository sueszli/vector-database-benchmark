from sortedcontainers import SortedList

class Solution(object):

    def getSubarrayBeauty(self, nums, k, x):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :type x: int\n        :rtype: List[int]\n        '
        result = []
        sl = SortedList()
        for (i, v) in enumerate(nums):
            if i - k >= 0:
                sl.remove(nums[i - k])
            sl.add(v)
            if i - k + 1 >= 0:
                result.append(min(sl[x - 1], 0))
        return result