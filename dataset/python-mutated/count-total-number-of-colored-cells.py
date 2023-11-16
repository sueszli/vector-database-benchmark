class Solution(object):

    def coloredCells(self, n):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :rtype: int\n        '
        return n ** 2 + (n - 1) ** 2

class Solution2(object):

    def coloredCells(self, n):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :rtype: int\n        '
        return (1 + (1 + 2 * (n - 1))) * n // 2 * 2 - (2 * n - 1)