import itertools

class Solution(object):

    def numTilings(self, N):
        if False:
            i = 10
            return i + 15
        '\n        :type N: int\n        :rtype: int\n        '
        M = int(1000000000.0 + 7)

        def matrix_expo(A, K):
            if False:
                while True:
                    i = 10
            result = [[int(i == j) for j in xrange(len(A))] for i in xrange(len(A))]
            while K:
                if K % 2:
                    result = matrix_mult(result, A)
                A = matrix_mult(A, A)
                K /= 2
            return result

        def matrix_mult(A, B):
            if False:
                return 10
            ZB = zip(*B)
            return [[sum((a * b for (a, b) in itertools.izip(row, col))) % M for col in ZB] for row in A]
        T = [[1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 0]]
        return matrix_mult([[1, 0, 0, 0]], matrix_expo(T, N))[0][0]

class Solution2(object):

    def numTilings(self, N):
        if False:
            i = 10
            return i + 15
        '\n        :type N: int\n        :rtype: int\n        '
        M = int(1000000000.0 + 7)
        dp = [1, 1, 2]
        for i in xrange(3, N + 1):
            dp[i % 3] = (2 * dp[(i - 1) % 3] % M + dp[(i - 3) % 3]) % M
        return dp[N % 3]