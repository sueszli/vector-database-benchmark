class Solution(object):

    def minimumSum(self, n, k):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type k: int\n        :rtype: int\n        '

        def arithmetic_progression_sum(a, d, n):
            if False:
                while True:
                    i = 10
            return (a + (a + (n - 1) * d)) * n // 2
        a = min(k // 2, n)
        b = n - a
        return arithmetic_progression_sum(1, 1, a) + arithmetic_progression_sum(k, 1, b)