class Solution(object):

    def splitArraySameAverage(self, A):
        if False:
            i = 10
            return i + 15
        '\n        :type A: List[int]\n        :rtype: bool\n        '

        def possible(total, n):
            if False:
                print('Hello World!')
            for i in xrange(1, n // 2 + 1):
                if total * i % n == 0:
                    return True
            return False
        (n, s) = (len(A), sum(A))
        if not possible(n, s):
            return False
        sums = [set() for _ in xrange(n // 2 + 1)]
        sums[0].add(0)
        for num in A:
            for i in reversed(xrange(1, n // 2 + 1)):
                for prev in sums[i - 1]:
                    sums[i].add(prev + num)
        for i in xrange(1, n // 2 + 1):
            if s * i % n == 0 and s * i // n in sums[i]:
                return True
        return False