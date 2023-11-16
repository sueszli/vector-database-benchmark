class Solution(object):

    def buyChoco(self, prices, money):
        if False:
            return 10
        '\n        :type prices: List[int]\n        :type money: int\n        :rtype: int\n        '
        i = min(xrange(len(prices)), key=lambda x: prices[x])
        j = min((j for j in xrange(len(prices)) if j != i), key=lambda x: prices[x])
        return money - (prices[i] + prices[j]) if prices[i] + prices[j] <= money else money