class Solution(object):

    def makeIntegerBeautiful(self, n, target):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type target: int\n        :rtype: int\n        '
        (total, m) = (0, n)
        while m:
            total += m % 10
            m //= 10
        (m, l) = (n, 0)
        while total > target:
            while True:
                total -= m % 10
                m //= 10
                l += 1
                if m % 10 != 9:
                    break
            total += 1
            m += 1
        return m * 10 ** l - n