class Solution(object):

    def minCost(self, houses, cost, m, n, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type houses: List[int]\n        :type cost: List[List[int]]\n        :type m: int\n        :type n: int\n        :type target: int\n        :rtype: int\n        '
        dp = [[[float('inf') for _ in xrange(n)] for _ in xrange(target)] for _ in xrange(2)]
        for i in xrange(m):
            dp[i % 2] = [[float('inf') for _ in xrange(n)] for _ in xrange(target)]
            for j in xrange(min(target, i + 1)):
                for k in xrange(n):
                    if houses[i] and houses[i] - 1 != k:
                        continue
                    same = dp[(i - 1) % 2][j][k] if i - 1 >= 0 else 0
                    diff = (min([dp[(i - 1) % 2][j - 1][nk] for nk in xrange(n) if nk != k] or [float('inf')]) if j - 1 >= 0 else float('inf')) if i - 1 >= 0 else 0
                    paint = cost[i][k] if not houses[i] else 0
                    dp[i % 2][j][k] = min(same, diff) + paint
        result = min(dp[(m - 1) % 2][-1])
        return result if result != float('inf') else -1

class Solution2(object):

    def minCost(self, houses, cost, m, n, target):
        if False:
            print('Hello World!')
        '\n        :type houses: List[int]\n        :type cost: List[List[int]]\n        :type m: int\n        :type n: int\n        :type target: int\n        :rtype: int\n        '
        dp = {(0, 0): 0}
        for (i, p) in enumerate(houses):
            new_dp = {}
            for nk in xrange(1, n + 1) if not p else [p]:
                for (j, k) in dp:
                    nj = j + (k != nk)
                    if nj > target:
                        continue
                    new_dp[nj, nk] = min(new_dp.get((nj, nk), float('inf')), dp[j, k] + (cost[i][nk - 1] if nk != p else 0))
            dp = new_dp
        return min([dp[j, k] for (j, k) in dp if j == target] or [-1])