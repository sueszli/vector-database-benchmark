class Solution(object):

    def minimumXORSum(self, nums1, nums2):
        if False:
            for i in range(10):
                print('nop')

        def hungarian(a):
            if False:
                i = 10
                return i + 15
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
        adj = [[0] * len(nums2) for _ in xrange(len(nums1))]
        for i in xrange(len(nums1)):
            for j in xrange(len(nums2)):
                adj[i][j] = nums1[i] ^ nums2[j]
        return hungarian(adj)[0]

class Solution2(object):

    def minimumXORSum(self, nums1, nums2):
        if False:
            while True:
                i = 10
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: int\n        '
        dp = [(float('inf'), float('inf'))] * 2 ** len(nums2)
        dp[0] = (0, 0)
        for mask in xrange(len(dp)):
            bit = 1
            for i in xrange(len(nums2)):
                if mask & bit == 0:
                    dp[mask | bit] = min(dp[mask | bit], (dp[mask][0] + (nums1[dp[mask][1]] ^ nums2[i]), dp[mask][1] + 1))
                bit <<= 1
        return dp[-1][0]