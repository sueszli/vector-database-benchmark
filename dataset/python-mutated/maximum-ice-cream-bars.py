class Solution(object):

    def maxIceCream(self, costs, coins):
        if False:
            return 10
        '\n        :type costs: List[int]\n        :type coins: int\n        :rtype: int\n        '
        costs.sort()
        for (i, c) in enumerate(costs):
            coins -= c
            if coins < 0:
                return i
        return len(costs)