class Solution(object):

    def alternateDigitSum(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: int\n        '
        result = 0
        sign = 1
        while n:
            sign *= -1
            result += sign * (n % 10)
            n //= 10
        return sign * result