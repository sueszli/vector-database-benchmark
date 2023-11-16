class Solution(object):

    def minimumMoves(self, grid):
        if False:
            while True:
                i = 10
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '

        def hungarian(a):
            if False:
                return 10
            if not a:
                return (0, [])
            (n, m) = (len(a) + 1, len(a[0]) + 1)
            (u, v, p, ans) = ([0] * n, [0] * m, [0] * m, [0] * (n - 1))
            for i in xrange(1, n):
                p[0] = i
                j0 = 0
                (dist, pre) = ([float('inf')] * m, [-1] * m)
                done = [False] * (m + 1)
                while True:
                    done[j0] = True
                    (i0, j1, delta) = (p[j0], None, float('inf'))
                    for j in xrange(1, m):
                        if done[j]:
                            continue
                        cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < dist[j]:
                            (dist[j], pre[j]) = (cur, j0)
                        if dist[j] < delta:
                            (delta, j1) = (dist[j], j)
                    for j in xrange(m):
                        if done[j]:
                            u[p[j]] += delta
                            v[j] -= delta
                        else:
                            dist[j] -= delta
                    j0 = j1
                    if not p[j0]:
                        break
                while j0:
                    j1 = pre[j0]
                    (p[j0], j0) = (p[j1], j1)
            for j in xrange(1, m):
                if p[j]:
                    ans[p[j] - 1] = j - 1
            return (-v[0], ans)

        def dist(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        (src, dst) = ([], [])
        for i in xrange(len(grid)):
            for j in xrange(len(grid[0])):
                if grid[i][j] - 1 >= 0:
                    src.extend([(i, j)] * (grid[i][j] - 1))
                else:
                    dst.append((i, j))
        adj = [[dist(src[i], dst[j]) for j in xrange(len(dst))] for i in xrange(len(src))]
        return hungarian(adj)[0]
from scipy.optimize import linear_sum_assignment as hungarian
import itertools

class Solution2(object):

    def minimumMoves(self, grid):
        if False:
            i = 10
            return i + 15
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '

        def dist(a, b):
            if False:
                print('Hello World!')
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        (src, dst) = ([], [])
        for i in xrange(len(grid)):
            for j in xrange(len(grid[0])):
                if grid[i][j] - 1 >= 0:
                    src.extend([(i, j)] * (grid[i][j] - 1))
                else:
                    dst.append((i, j))
        adj = [[dist(src[i], dst[j]) for j in xrange(len(dst))] for i in xrange(len(src))]
        return sum((adj[i][j] for (i, j) in itertools.izip(*hungarian(adj))))

class Solution3(object):

    def minimumMoves(self, grid):
        if False:
            i = 10
            return i + 15
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '

        def dist(a, b):
            if False:
                while True:
                    i = 10
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def backtracking(curr):
            if False:
                while True:
                    i = 10
            if curr == len(zero):
                return 0
            result = float('inf')
            (i, j) = zero[curr]
            for ni in xrange(len(grid)):
                for nj in xrange(len(grid[0])):
                    if not grid[ni][nj] >= 2:
                        continue
                    grid[ni][nj] -= 1
                    result = min(result, dist((i, j), (ni, nj)) + backtracking(curr + 1))
                    grid[ni][nj] += 1
            return result
        zero = [(i, j) for i in xrange(len(grid)) for j in xrange(len(grid[0])) if grid[i][j] == 0]
        return backtracking(0)