class Solution(object):

    def numberOfWays(self, n, m, k, source, dest):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type m: int\n        :type k: int\n        :type source: List[int]\n        :type dest: List[int]\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7

        def matrix_mult(A, B):
            if False:
                print('Hello World!')
            ZB = zip(*B)
            return [[sum((a * b % MOD for (a, b) in itertools.izip(row, col))) % MOD for col in ZB] for row in A]

        def matrix_expo(A, K):
            if False:
                i = 10
                return i + 15
            result = [[int(i == j) for j in xrange(len(A))] for i in xrange(len(A))]
            while K:
                if K % 2:
                    result = matrix_mult(result, A)
                A = matrix_mult(A, A)
                K /= 2
            return result
        T = [[0, m - 1, n - 1, 0], [1, m - 2, 0, n - 1], [1, 0, n - 2, m - 1], [0, 1, 1, n - 2 + (m - 2)]]
        dp = [0] * 4
        if source == dest:
            dp[0] = 1
        elif source[0] == dest[0]:
            dp[1] = 1
        elif source[1] == dest[1]:
            dp[2] = 1
        else:
            dp[3] = 1
        dp = matrix_mult([dp], matrix_expo(T, k))[0]
        return dp[0]

class Solution2(object):

    def numberOfWays(self, n, m, k, source, dest):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type m: int\n        :type k: int\n        :type source: List[int]\n        :type dest: List[int]\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        both_same = row_same = col_same = no_same = 0
        if source == dest:
            both_same = 1
        elif source[0] == dest[0]:
            row_same = 1
        elif source[1] == dest[1]:
            col_same = 1
        else:
            no_same = 1
        for _ in xrange(k):
            (both_same, row_same, col_same, no_same) = ((row_same + col_same) % MOD, (both_same * (m - 1) + row_same * (m - 2) + no_same) % MOD, (both_same * (n - 1) + col_same * (n - 2) + no_same) % MOD, (row_same * (n - 1) + col_same * (m - 1) + no_same * (n - 2 + (m - 2))) % MOD)
        return both_same