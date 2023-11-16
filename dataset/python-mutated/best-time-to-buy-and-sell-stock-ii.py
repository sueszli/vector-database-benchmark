class Solution(object):

    def maxProfit(self, prices):
        if False:
            for i in range(10):
                print('nop')
        profit = 0
        for i in xrange(len(prices) - 1):
            profit += max(0, prices[i + 1] - prices[i])
        return profit

    def maxProfit2(self, prices):
        if False:
            while True:
                i = 10
        return sum(map(lambda x: max(prices[x + 1] - prices[x], 0), xrange(len(prices[:-1]))))