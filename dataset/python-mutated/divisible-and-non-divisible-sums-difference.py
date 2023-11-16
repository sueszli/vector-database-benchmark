class Solution(object):

    def differenceOfSums(self, n, m):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type m: int\n        :rtype: int\n        '

        def arithmetic_progression_sum(a, d, l):
            if False:
                for i in range(10):
                    print('nop')
            return (a + (a + (l - 1) * d)) * l // 2
        return arithmetic_progression_sum(1, 1, n) - 2 * arithmetic_progression_sum(m, m, n // m)

class Solution2(object):

    def differenceOfSums(self, n, m):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type m: int\n        :rtype: int\n        '
        return (n + 1) * n // 2 - 2 * ((n // m + 1) * (n // m) // 2 * m)