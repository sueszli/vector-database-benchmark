import collections

class Solution(object):

    def countSubgraphsForEachDiameter(self, n, edges):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: List[int]\n        '

        def dfs(n, adj, curr, parent, lookup, count, dp):
            if False:
                i = 10
                return i + 15
            for child in adj[curr]:
                if child == parent or lookup[child]:
                    continue
                dfs(n, adj, child, curr, lookup, count, dp)
            dp[curr][0][0] = 1
            for child in adj[curr]:
                if child == parent or lookup[child]:
                    continue
                new_dp_curr = [row[:] for row in dp[curr]]
                for curr_d in xrange(count[curr]):
                    for curr_max_d in xrange(curr_d, min(2 * curr_d + 1, count[curr])):
                        if not dp[curr][curr_d][curr_max_d]:
                            continue
                        for child_d in xrange(count[child]):
                            for child_max_d in xrange(child_d, min(2 * child_d + 1, count[child])):
                                new_dp_curr[max(curr_d, child_d + 1)][max(curr_max_d, child_max_d, curr_d + child_d + 1)] += dp[curr][curr_d][curr_max_d] * dp[child][child_d][child_max_d]
                count[curr] += count[child]
                dp[curr] = new_dp_curr
        adj = collections.defaultdict(list)
        for (u, v) in edges:
            u -= 1
            v -= 1
            adj[u].append(v)
            adj[v].append(u)
        (lookup, result) = ([0] * n, [0] * (n - 1))
        for i in xrange(n):
            dp = [[[0] * n for _ in xrange(n)] for _ in xrange(n)]
            count = [1] * n
            dfs(n, adj, i, -1, lookup, count, dp)
            lookup[i] = 1
            for d in xrange(1, n):
                for max_d in xrange(d, min(2 * d + 1, n)):
                    result[max_d - 1] += dp[i][d][max_d]
        return result
import collections
import math

class Solution2(object):

    def countSubgraphsForEachDiameter(self, n, edges):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: List[int]\n        '

        def popcount(mask):
            if False:
                return 10
            count = 0
            while mask:
                mask &= mask - 1
                count += 1
            return count

        def bfs(adj, mask, start):
            if False:
                i = 10
                return i + 15
            q = collections.deque([(start, 0)])
            lookup = 1 << start
            count = popcount(mask) - 1
            (u, d) = (None, None)
            while q:
                (u, d) = q.popleft()
                for v in adj[u]:
                    if not mask & 1 << v or lookup & 1 << v:
                        continue
                    lookup |= 1 << v
                    count -= 1
                    q.append((v, d + 1))
            return (count == 0, u, d)

        def max_distance(n, edges, adj, mask):
            if False:
                for i in range(10):
                    print('nop')
            (is_valid, farthest, _) = bfs(adj, mask, int(math.log(mask & -mask, 2)))
            return bfs(adj, mask, farthest)[-1] if is_valid else 0
        adj = collections.defaultdict(list)
        for (u, v) in edges:
            u -= 1
            v -= 1
            adj[u].append(v)
            adj[v].append(u)
        result = [0] * (n - 1)
        for mask in xrange(1, 2 ** n):
            max_d = max_distance(n, edges, adj, mask)
            if max_d - 1 >= 0:
                result[max_d - 1] += 1
        return result