class Solution(object):

    def countNumbersWithUniqueDigits(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: int\n        '
        if n == 0:
            return 1
        (count, fk) = (10, 9)
        for k in xrange(2, n + 1):
            fk *= 10 - (k - 1)
            count += fk
        return count