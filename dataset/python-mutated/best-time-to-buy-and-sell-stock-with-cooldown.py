class Solution(object):

    def maxProfit(self, prices):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type prices: List[int]\n        :rtype: int\n        '
        if not prices:
            return 0
        (buy, sell, coolDown) = ([0] * 2, [0] * 2, [0] * 2)
        buy[0] = -prices[0]
        for i in xrange(1, len(prices)):
            buy[i % 2] = max(buy[(i - 1) % 2], coolDown[(i - 1) % 2] - prices[i])
            sell[i % 2] = buy[(i - 1) % 2] + prices[i]
            coolDown[i % 2] = max(coolDown[(i - 1) % 2], sell[(i - 1) % 2])
        return max(coolDown[(len(prices) - 1) % 2], sell[(len(prices) - 1) % 2])