class Solution(object):

    def minimumPossibleSum(self, n, target):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type target: int\n        :rtype: int\n        '

        def arithmetic_progression_sum(a, d, n):
            if False:
                for i in range(10):
                    print('nop')
            return (a + (a + (n - 1) * d)) * n // 2
        a = min(target // 2, n)
        b = n - a
        return arithmetic_progression_sum(1, 1, a) + arithmetic_progression_sum(target, 1, b)