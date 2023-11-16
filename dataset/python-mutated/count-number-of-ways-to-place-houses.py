import itertools

class Solution(object):

    def countHousePlacements(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7

        def matrix_mult(A, B):
            if False:
                for i in range(10):
                    print('nop')
            ZB = zip(*B)
            return [[sum((a * b % MOD for (a, b) in itertools.izip(row, col))) % MOD for col in ZB] for row in A]

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
        T = [[1, 1], [1, 0]]
        return pow(matrix_mult([[1, 0]], matrix_expo(T, n + 1))[0][0], 2, MOD)

class Solution2(object):

    def countHousePlacements(self, n):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        (prev, curr) = (0, 1)
        for _ in xrange(n + 1):
            (prev, curr) = (curr, (prev + curr) % MOD)
        return pow(curr, 2, MOD)