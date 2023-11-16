class Solution(object):

    def getDescentPeriods(self, prices):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type prices: List[int]\n        :rtype: int\n        '
        result = l = 0
        for i in xrange(len(prices)):
            l += 1
            if i + 1 == len(prices) or prices[i] - 1 != prices[i + 1]:
                result += l * (l + 1) // 2
                l = 0
        return result