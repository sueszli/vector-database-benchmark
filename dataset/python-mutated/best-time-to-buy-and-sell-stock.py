class Solution(object):

    def maxProfit(self, prices):
        if False:
            while True:
                i = 10
        (max_profit, min_price) = (0, float('inf'))
        for price in prices:
            min_price = min(min_price, price)
            max_profit = max(max_profit, price - min_price)
        return max_profit