class Solution(object):

    def flipLights(self, n, m):
        if False:
            return 10
        '\n        :type n: int\n        :type m: int\n        :rtype: int\n        '
        if m == 0:
            return 1
        if n == 1:
            return 2
        if m == 1 and n == 2:
            return 3
        if m == 1 or n == 2:
            return 4
        if m == 2:
            return 7
        return 8