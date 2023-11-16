import itertools

class Solution(object):

    def countVowelPermutation(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: int\n        '

        def matrix_expo(A, K):
            if False:
                for i in range(10):
                    print('nop')
            result = [[int(i == j) for j in xrange(len(A))] for i in xrange(len(A))]
            while K:
                if K % 2:
                    result = matrix_mult(result, A)
                A = matrix_mult(A, A)
                K /= 2
            return result

        def matrix_mult(A, B):
            if False:
                print('Hello World!')
            ZB = zip(*B)
            return [[sum((a * b for (a, b) in itertools.izip(row, col))) % MOD for col in ZB] for row in A]
        MOD = 10 ** 9 + 7
        T = [[0, 1, 1, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 1, 1, 0]]
        return sum(map(sum, matrix_expo(T, n - 1))) % MOD

class Solution2(object):

    def countVowelPermutation(self, n):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        (a, e, i, o, u) = (1, 1, 1, 1, 1)
        for _ in xrange(1, n):
            (a, e, i, o, u) = ((e + i + u) % MOD, (a + i) % MOD, (e + o) % MOD, i, (i + o) % MOD)
        return (a + e + i + o + u) % MOD