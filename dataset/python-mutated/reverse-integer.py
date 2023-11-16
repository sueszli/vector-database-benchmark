class Solution(object):

    def reverse(self, x):
        if False:
            while True:
                i = 10
        '\n        :type x: int\n        :rtype: int\n        '
        if x < 0:
            return -self.reverse(-x)
        result = 0
        while x:
            result = result * 10 + x % 10
            x //= 10
        return result if result <= 2147483647 else 0

    def reverse2(self, x):
        if False:
            while True:
                i = 10
        '\n        :type x: int\n        :rtype: int\n        '
        if x < 0:
            x = int(str(x)[::-1][-1] + str(x)[::-1][:-1])
        else:
            x = int(str(x)[::-1])
        x = 0 if abs(x) > 2147483647 else x
        return x

    def reverse3(self, x):
        if False:
            return 10
        '\n        :type x: int\n        :rtype: int\n        '
        s = cmp(x, 0)
        r = int(repr(s * x)[::-1])
        return s * r * (r < 2 ** 31)