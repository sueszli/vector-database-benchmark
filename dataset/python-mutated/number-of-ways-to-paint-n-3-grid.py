import itertools

class Solution(object):

    def numOfWays(self, n):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :rtype: int\n        '

        def matrix_expo(A, K):
            if False:
                print('Hello World!')
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
            return [[sum((a * b % MOD for (a, b) in itertools.izip(row, col))) % MOD for col in ZB] for row in A]
        MOD = 10 ** 9 + 7
        T = [[3, 2], [2, 2]]
        return sum(matrix_mult([[6, 6]], matrix_expo(T, n - 1))[0]) % MOD

class Solution2(object):

    def numOfWays(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        (aba, abc) = (6, 6)
        for _ in xrange(n - 1):
            (aba, abc) = ((3 * aba % MOD + 2 * abc % MOD) % MOD, (2 * abc % MOD + 2 * aba % MOD) % MOD)
        return (aba + abc) % MOD