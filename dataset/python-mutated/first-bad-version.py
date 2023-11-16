class Solution(object):

    def firstBadVersion(self, n):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :rtype: int\n        '
        (left, right) = (1, n)
        while left <= right:
            mid = left + (right - left) / 2
            if isBadVersion(mid):
                right = mid - 1
            else:
                left = mid + 1
        return left