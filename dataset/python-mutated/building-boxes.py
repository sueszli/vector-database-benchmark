import math

class Solution(object):

    def minimumBoxes(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: int\n        '
        h = int((6 * n) ** (1.0 / 3))
        if h * (h + 1) * (h + 2) > 6 * n:
            h -= 1
        n -= h * (h + 1) * (h + 2) // 6
        d = int(math.ceil((-1 + (1 + 8 * n) ** 0.5) / 2))
        return h * (h + 1) // 2 + d