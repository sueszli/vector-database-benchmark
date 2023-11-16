class Solution(object):

    def finalPrices(self, prices):
        if False:
            i = 10
            return i + 15
        '\n        :type prices: List[int]\n        :rtype: List[int]\n        '
        stk = []
        for (i, p) in enumerate(prices):
            while stk and prices[stk[-1]] >= p:
                prices[stk.pop()] -= p
            stk.append(i)
        return prices