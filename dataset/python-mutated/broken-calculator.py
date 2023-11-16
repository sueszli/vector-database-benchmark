class Solution(object):

    def brokenCalc(self, X, Y):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type X: int\n        :type Y: int\n        :rtype: int\n        '
        result = 0
        while X < Y:
            if Y % 2:
                Y += 1
            else:
                Y /= 2
            result += 1
        return result + X - Y