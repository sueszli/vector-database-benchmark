class Solution(object):

    def minimumChanges(self, s, k):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :type k: int\n        :rtype: int\n        '
        divisors = [[] for _ in xrange(len(s) + 1)]
        for i in xrange(1, len(divisors)):
            for j in xrange(i, len(divisors), i):
                divisors[j].append(i)
        dp = [[{} for _ in xrange(len(s))] for _ in xrange(len(s))]
        for l in xrange(1, len(s) + 1):
            for left in xrange(len(s) - l + 1):
                right = left + l - 1
                for d in divisors[l]:
                    dp[left][right][d] = (dp[left + d][right - d][d] if left + d < right - d else 0) + sum((s[left + i] != s[right - (d - 1) + i] for i in xrange(d)))
        dp2 = [[min((dp[i][j][d] for d in divisors[j - i + 1] if d != j - i + 1)) if i < j else 0 for j in xrange(len(s))] for i in xrange(len(s))]
        dp3 = [len(s)] * (len(s) + 1)
        dp3[0] = 0
        for l in xrange(k):
            new_dp3 = [len(s)] * (len(s) + 1)
            for i in xrange(len(s)):
                for j in xrange(l * 2, i):
                    new_dp3[i + 1] = min(new_dp3[i + 1], dp3[j] + dp2[j][i])
            dp3 = new_dp3
        return dp3[len(s)]

class Solution2(object):

    def minimumChanges(self, s, k):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :type k: int\n        :rtype: int\n        '
        divisors = [[] for _ in xrange(len(s) + 1)]
        for i in xrange(1, len(divisors)):
            for j in xrange(i, len(divisors), i):
                divisors[j].append(i)
        dp = [[{} for _ in xrange(len(s))] for _ in xrange(len(s))]
        for l in xrange(1, len(s) + 1):
            for left in xrange(len(s) - l + 1):
                right = left + l - 1
                for d in divisors[l]:
                    dp[left][right][d] = (dp[left + d][right - d][d] if left + d < right - d else 0) + sum((s[left + i] != s[right - (d - 1) + i] for i in xrange(d)))
        dp2 = [[len(s)] * (k + 1) for _ in xrange(len(s) + 1)]
        dp2[0][0] = 0
        for i in xrange(len(s)):
            for j in xrange(i):
                c = min((dp[j][i][d] for d in divisors[i - j + 1] if d != i - j + 1))
                for l in xrange(k):
                    dp2[i + 1][l + 1] = min(dp2[i + 1][l + 1], dp2[j][l] + c)
        return dp2[len(s)][k]

class Solution3(object):

    def minimumChanges(self, s, k):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :type k: int\n        :rtype: int\n        '

        def min_dist(left, right):
            if False:
                print('Hello World!')
            return min((sum((s[left + i] != s[right - ((i // d + 1) * d - 1) + i % d] for i in xrange((right - left + 1) // 2))) for d in divisors[right - left + 1]))
        divisors = [[] for _ in xrange(len(s) + 1)]
        for i in xrange(1, len(divisors)):
            for j in xrange(i + i, len(divisors), i):
                divisors[j].append(i)
        dp = [[len(s)] * (k + 1) for _ in xrange(len(s) + 1)]
        dp[0][0] = 0
        for i in xrange(len(s)):
            for j in xrange(i):
                c = min_dist(j, i)
                for l in xrange(k):
                    dp[i + 1][l + 1] = min(dp[i + 1][l + 1], dp[j][l] + c)
        return dp[len(s)][k]