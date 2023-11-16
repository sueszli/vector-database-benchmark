import itertools

class Solution(object):

    def maxCompatibilitySum(self, students, mentors):
        if False:
            print('Hello World!')
        '\n        :type students: List[List[int]]\n        :type mentors: List[List[int]]\n        :rtype: int\n        '

        def hungarian(a):
            if False:
                print('Hello World!')
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

        def score(s, m):
            if False:
                for i in range(10):
                    print('nop')
            return sum((int(a == b) for (a, b) in itertools.izip(s, m)))
        return -hungarian([[-score(s, m) for m in mentors] for s in students])[0]

class Solution2(object):

    def maxCompatibilitySum(self, students, mentors):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type students: List[List[int]]\n        :type mentors: List[List[int]]\n        :rtype: int\n        '

        def popcount(n):
            if False:
                print('Hello World!')
            result = 0
            while n:
                n &= n - 1
                result += 1
            return result

        def masks(vvi):
            if False:
                i = 10
                return i + 15
            result = []
            for vi in vvi:
                (mask, bit) = (0, 1)
                for i in xrange(len(vi)):
                    if vi[i]:
                        mask |= bit
                    bit <<= 1
                result.append(mask)
            return result
        (nums1, nums2) = (masks(students), masks(mentors))
        dp = [(0, 0)] * 2 ** len(nums2)
        for mask in xrange(len(dp)):
            bit = 1
            for i in xrange(len(nums2)):
                if mask & bit == 0:
                    dp[mask | bit] = max(dp[mask | bit], (dp[mask][0] + (len(students[0]) - popcount(nums1[dp[mask][1]] ^ nums2[i])), dp[mask][1] + 1))
                bit <<= 1
        return dp[-1][0]