class Solution(object):

    def nextGreaterElement(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: int\n        '
        digits = map(int, list(str(n)))
        (k, l) = (-1, 0)
        for i in xrange(len(digits) - 1):
            if digits[i] < digits[i + 1]:
                k = i
        if k == -1:
            digits.reverse()
            return -1
        for i in xrange(k + 1, len(digits)):
            if digits[i] > digits[k]:
                l = i
        (digits[k], digits[l]) = (digits[l], digits[k])
        digits[k + 1:] = digits[:k:-1]
        result = int(''.join(map(str, digits)))
        return -1 if result >= 2147483647 else result