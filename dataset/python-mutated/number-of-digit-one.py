class Solution(object):

    def countDigitOne(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: int\n        '
        DIGIT = 1
        is_zero = int(DIGIT == 0)
        result = is_zero
        base = 1
        while n >= base:
            result += (n // (10 * base) - is_zero) * base + min(base, max(n % (10 * base) - DIGIT * base + 1, 0))
            base *= 10
        return result