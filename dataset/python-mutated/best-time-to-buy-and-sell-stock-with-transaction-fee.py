class Solution(object):

    def maxProfit(self, prices, fee):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type prices: List[int]\n        :type fee: int\n        :rtype: int\n        '
        (cash, hold) = (0, -prices[0])
        for i in xrange(1, len(prices)):
            cash = max(cash, hold + prices[i] - fee)
            hold = max(hold, cash - prices[i])
        return cash