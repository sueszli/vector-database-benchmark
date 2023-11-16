import itertools

class Solution(object):

    def fib(self, N):
        if False:
            print('Hello World!')
        '\n        :type N: int\n        :rtype: int\n        '

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
                for i in range(10):
                    print('nop')
            ZB = zip(*B)
            return [[sum((a * b for (a, b) in itertools.izip(row, col))) for col in ZB] for row in A]
        T = [[1, 1], [1, 0]]
        return matrix_mult([[1, 0]], matrix_expo(T, N))[0][1]

class Solution2(object):

    def fib(self, N):
        if False:
            i = 10
            return i + 15
        '\n        :type N: int\n        :rtype: int\n        '
        (prev, current) = (0, 1)
        for i in xrange(N):
            (prev, current) = (current, prev + current)
        return prev