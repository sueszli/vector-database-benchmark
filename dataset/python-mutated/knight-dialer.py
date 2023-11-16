import itertools

class Solution(object):

    def knightDialer(self, N):
        if False:
            while True:
                i = 10
        '\n        :type N: int\n        :rtype: int\n        '

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

        def matrix_mult(A, B):
            if False:
                for i in range(10):
                    print('nop')
            ZB = zip(*B)
            return [[sum((a * b for (a, b) in itertools.izip(row, col))) % M for col in ZB] for row in A]
        M = 10 ** 9 + 7
        T = [[0, 0, 0, 0, 1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0, 0]]
        return sum(map(sum, matrix_expo(T, N - 1))) % M

class Solution2(object):

    def knightDialer(self, N):
        if False:
            return 10
        '\n        :type N: int\n        :rtype: int\n        '
        M = 10 ** 9 + 7
        moves = [[4, 6], [6, 8], [7, 9], [4, 8], [3, 9, 0], [], [1, 7, 0], [2, 6], [1, 3], [2, 4]]
        dp = [[1 for _ in xrange(10)] for _ in xrange(2)]
        for i in xrange(N - 1):
            dp[(i + 1) % 2] = [0] * 10
            for j in xrange(10):
                for nei in moves[j]:
                    dp[(i + 1) % 2][nei] += dp[i % 2][j]
                    dp[(i + 1) % 2][nei] %= M
        return sum(dp[(N - 1) % 2]) % M