import itertools

class Solution(object):

    def climbStairs(self, n):
        if False:
            print('Hello World!')
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
                while True:
                    i = 10
            ZB = zip(*B)
            return [[sum((a * b for (a, b) in itertools.izip(row, col))) for col in ZB] for row in A]
        T = [[1, 1], [1, 0]]
        return matrix_mult([[1, 0]], matrix_expo(T, n))[0][0]

class Solution2(object):
    """
    :type n: int
    :rtype: int
    """

    def climbStairs(self, n):
        if False:
            while True:
                i = 10
        (prev, current) = (0, 1)
        for i in xrange(n):
            (prev, current) = (current, prev + current)
        return current