class Solution(object):

    def passThePillow(self, n, time):
        if False:
            return 10
        '\n        :type n: int\n        :type time: int\n        :rtype: int\n        '
        return n - abs(n - 1 - time % (2 * (n - 1)))