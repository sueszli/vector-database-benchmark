class Solution(object):

    def newInteger(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: int\n        '
        (result, base) = (0, 1)
        while n > 0:
            result += n % 9 * base
            n /= 9
            base *= 10
        return result