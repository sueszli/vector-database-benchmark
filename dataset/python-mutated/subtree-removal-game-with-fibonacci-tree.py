class Solution(object):

    def findGameWinner(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: bool\n        '
        return n % 6 != 1

class Solution2(object):

    def findGameWinner(self, n):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :rtype: bool\n        '
        grundy = [0, 1]
        for i in xrange(2, n):
            grundy[i % 2] = grundy[(i - 1) % 2] + 1 ^ grundy[(i - 2) % 2] + 1
        return grundy[(n - 1) % 2] > 0