class Solution(object):

    def isReachable(self, targetX, targetY):
        if False:
            i = 10
            return i + 15
        '\n        :type targetX: int\n        :type targetY: int\n        :rtype: bool\n        '

        def gcd(a, b):
            if False:
                print('Hello World!')
            while b:
                (a, b) = (b, a % b)
            return a
        g = gcd(targetX, targetY)
        return g == g & ~(g - 1)