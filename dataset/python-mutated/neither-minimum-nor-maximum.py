class Solution(object):

    def findNonMinOrMax(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        (mx, mn) = (float('-inf'), float('inf'))
        result = -1
        for x in nums:
            if mn < x < mx:
                return x
            if x < mn:
                result = mn
                mn = x
            if x > mx:
                result = mx
                mx = x
        return result if mn < result < mx else -1

class Solution2(object):

    def findNonMinOrMax(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        (mx, mn) = (max(nums), min(nums))
        return next((x for x in nums if x not in (mx, mn)), -1)