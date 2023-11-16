class Solution(object):

    def divide(self, dividend, divisor):
        if False:
            return 10
        '\n        :type dividend: int\n        :type divisor: int\n        :rtype: int\n        '
        (result, dvd, dvs) = (0, abs(dividend), abs(divisor))
        while dvd >= dvs:
            inc = dvs
            i = 0
            while dvd >= inc:
                dvd -= inc
                result += 1 << i
                inc <<= 1
                i += 1
        if dividend > 0 and divisor < 0 or (dividend < 0 and divisor > 0):
            return -result
        else:
            return result

    def divide2(self, dividend, divisor):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type dividend: int\n        :type divisor: int\n        :rtype: int\n        '
        positive = (dividend < 0) is (divisor < 0)
        (dividend, divisor) = (abs(dividend), abs(divisor))
        res = 0
        while dividend >= divisor:
            (temp, i) = (divisor, 1)
            while dividend >= temp:
                dividend -= temp
                res += i
                i <<= 1
                temp <<= 1
        if not positive:
            res = -res
        return min(max(-2147483648, res), 2147483647)