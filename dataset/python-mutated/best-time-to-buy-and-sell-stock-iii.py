class Solution(object):

    def maxProfit(self, prices):
        if False:
            return 10
        '\n        :type prices: List[int]\n        :rtype: int\n        '
        (hold1, hold2) = (float('-inf'), float('-inf'))
        (release1, release2) = (0, 0)
        for i in prices:
            release1 = max(release1, hold1 + i)
            hold1 = max(hold1, -i)
            release2 = max(release2, hold2 + i)
            hold2 = max(hold2, release1 - i)
        return release2

class Solution2(object):

    def maxProfit(self, prices):
        if False:
            while True:
                i = 10
        '\n        :type prices: List[int]\n        :rtype: int\n        '

        def maxAtMostKPairsProfit(prices, k):
            if False:
                print('Hello World!')
            max_buy = [float('-inf') for _ in xrange(k + 1)]
            max_sell = [0 for _ in xrange(k + 1)]
            for i in xrange(len(prices)):
                for j in xrange(1, k + 1):
                    max_buy[j] = max(max_buy[j], max_sell[j - 1] - prices[i])
                    max_sell[j] = max(max_sell[j], max_buy[j] + prices[i])
            return max_sell[k]
        return maxAtMostKPairsProfit(prices, 2)

class Solution3(object):

    def maxProfit(self, prices):
        if False:
            return 10
        (min_price, max_profit_from_left, max_profits_from_left) = (float('inf'), 0, [])
        for price in prices:
            min_price = min(min_price, price)
            max_profit_from_left = max(max_profit_from_left, price - min_price)
            max_profits_from_left.append(max_profit_from_left)
        (max_price, max_profit_from_right, max_profits_from_right) = (0, 0, [])
        for i in reversed(range(len(prices))):
            max_price = max(max_price, prices[i])
            max_profit_from_right = max(max_profit_from_right, max_price - prices[i])
            max_profits_from_right.insert(0, max_profit_from_right)
        max_profit = 0
        for i in range(len(prices)):
            max_profit = max(max_profit, max_profits_from_left[i] + max_profits_from_right[i])
        return max_profit