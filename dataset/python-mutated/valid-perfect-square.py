class Solution(object):

    def isPerfectSquare(self, num):
        if False:
            while True:
                i = 10
        '\n        :type num: int\n        :rtype: bool\n        '
        (left, right) = (1, num)
        while left <= right:
            mid = left + (right - left) / 2
            if mid >= num / mid:
                right = mid - 1
            else:
                left = mid + 1
        return left == num / left and num % left == 0