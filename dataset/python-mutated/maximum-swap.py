class Solution(object):

    def maximumSwap(self, num):
        if False:
            while True:
                i = 10
        '\n        :type num: int\n        :rtype: int\n        '
        digits = list(str(num))
        (left, right) = (0, 0)
        max_idx = len(digits) - 1
        for i in reversed(xrange(len(digits))):
            if digits[i] > digits[max_idx]:
                max_idx = i
            elif digits[max_idx] > digits[i]:
                (left, right) = (i, max_idx)
        (digits[left], digits[right]) = (digits[right], digits[left])
        return int(''.join(digits))