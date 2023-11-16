import itertools

class Solution(object):

    def minimumCosts(self, regular, express, expressCost):
        if False:
            while True:
                i = 10
        '\n        :type regular: List[int]\n        :type express: List[int]\n        :type expressCost: int\n        :rtype: List[int]\n        '
        result = []
        dp = [0, expressCost]
        for (r, e) in itertools.izip(regular, express):
            dp = [min(dp[0] + r, dp[1] + e), min(dp[0] + (r + expressCost), dp[1] + e)]
            result.append(min(dp[0], dp[1]))
        return result