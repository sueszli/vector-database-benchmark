class Solution(object):

    def minCostSetTime(self, startAt, moveCost, pushCost, targetSeconds):
        if False:
            print('Hello World!')
        '\n        :type startAt: int\n        :type moveCost: int\n        :type pushCost: int\n        :type targetSeconds: int\n        :rtype: int\n        '

        def cost(m, s):
            if False:
                while True:
                    i = 10
            if not (0 <= m <= 99 and s <= 99):
                return float('inf')
            result = 0
            curr = startAt
            for x in map(int, list(str(m * 100 + s))):
                result += (moveCost if x != curr else 0) + pushCost
                curr = x
            return result
        (m, s) = divmod(targetSeconds, 60)
        return min(cost(m, s), cost(m - 1, s + 60))