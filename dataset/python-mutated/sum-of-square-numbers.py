import math

class Solution(object):

    def judgeSquareSum(self, c):
        if False:
            while True:
                i = 10
        '\n        :type c: int\n        :rtype: bool\n        '
        for a in xrange(int(math.sqrt(c)) + 1):
            b = int(math.sqrt(c - a ** 2))
            if a ** 2 + b ** 2 == c:
                return True
        return False