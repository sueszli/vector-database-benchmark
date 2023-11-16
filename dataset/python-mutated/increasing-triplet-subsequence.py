import bisect

class Solution(object):

    def increasingTriplet(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: bool\n        '
        (min_num, a, b) = (float('inf'), float('inf'), float('inf'))
        for c in nums:
            if min_num >= c:
                min_num = c
            elif b >= c:
                (a, b) = (min_num, c)
            else:
                return True
        return False

class Solution_Generalization(object):

    def increasingTriplet(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: bool\n        '

        def increasingKUplet(nums, k):
            if False:
                return 10
            inc = [float('inf')] * (k - 1)
            for num in nums:
                i = bisect.bisect_left(inc, num)
                if i >= k - 1:
                    return True
                inc[i] = num
            return k == 0
        return increasingKUplet(nums, 3)