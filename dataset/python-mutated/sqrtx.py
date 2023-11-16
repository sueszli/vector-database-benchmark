class Solution(object):

    def mySqrt(self, x):
        if False:
            print('Hello World!')
        '\n        :type x: int\n        :rtype: int\n        '
        if x < 2:
            return x
        (left, right) = (1, x // 2)
        while left <= right:
            mid = left + (right - left) // 2
            if mid > x / mid:
                right = mid - 1
            else:
                left = mid + 1
        return left - 1