class Solution(object):

    def trailingZeroes(self, n):
        if False:
            i = 10
            return i + 15
        result = 0
        while n > 0:
            result += n / 5
            n /= 5
        return result