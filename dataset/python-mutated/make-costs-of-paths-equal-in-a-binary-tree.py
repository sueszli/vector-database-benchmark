class Solution(object):

    def minIncrements(self, n, cost):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type cost: List[int]\n        :rtype: int\n        '
        result = 0
        for i in reversed(xrange(n // 2)):
            result += abs(cost[2 * i + 1] - cost[2 * i + 2])
            cost[i] += max(cost[2 * i + 1], cost[2 * i + 2])
        return result