class Solution(object):

    def guessNumber(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: int\n        '
        (left, right) = (1, n)
        while left <= right:
            mid = left + (right - left) / 2
            if guess(mid) <= 0:
                right = mid - 1
            else:
                left = mid + 1
        return left