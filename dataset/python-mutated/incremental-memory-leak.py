class Solution(object):

    def memLeak(self, memory1, memory2):
        if False:
            print('Hello World!')
        '\n        :type memory1: int\n        :type memory2: int\n        :rtype: List[int]\n        '

        def s(a, d, n):
            if False:
                return 10
            return (2 * a + (n - 1) * d) * n // 2

        def f(a, d, x):
            if False:
                i = 10
                return i + 15
            r = int((-(2 * a - d) + ((2 * a - d) ** 2 + 8 * d * x) ** 0.5) / (2 * d))
            if s(a, d, r) > x:
                r -= 1
            return r
        is_swapped = False
        if memory1 < memory2:
            (memory1, memory2) = (memory2, memory1)
            is_swapped = True
        n = f(1, 1, memory1 - memory2)
        memory1 -= s(1, 1, n)
        if memory1 == memory2:
            is_swapped = False
        l = f(n + 1, 2, memory1)
        r = f(n + 2, 2, memory2)
        memory1 -= s(n + 1, 2, l)
        memory2 -= s(n + 2, 2, r)
        if is_swapped:
            (memory1, memory2) = (memory2, memory1)
        return [n + l + r + 1, memory1, memory2]