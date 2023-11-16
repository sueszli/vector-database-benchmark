class Solution(object):

    def makeTheIntegerZero(self, num1, num2):
        if False:
            print('Hello World!')
        '\n        :type num1: int\n        :type num2: int\n        :rtype: int\n        '

        def popcount(x):
            if False:
                return 10
            result = 0
            while x:
                x &= x - 1
                result += 1
            return result
        for i in xrange(1, 60 + 1):
            if num1 - i * num2 < 0:
                break
            if popcount(num1 - i * num2) <= i <= num1 - i * num2:
                return i
        return -1