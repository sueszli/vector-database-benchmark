class Solution(object):

    def reinitializePermutation(self, n):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :rtype: int\n        '

        def discrete_log(a, b, m):
            if False:
                return 10
            a %= m
            b %= m
            n = int(m ** 0.5) + 1
            an = pow(a, n, m)
            vals = {}
            curr = b
            for q in xrange(n + 1):
                vals[curr] = q
                curr = curr * a % m
            curr = 1
            for p in xrange(1, n + 1):
                curr = curr * an % m
                if curr in vals:
                    return n * p - vals[curr]
            return -1
        return 1 + discrete_log(2, n // 2, n - 1)

class Solution2(object):

    def reinitializePermutation(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: int\n        '
        if n == 2:
            return 1
        (result, i) = (0, 1)
        while not result or i != 1:
            i = i * 2 % (n - 1)
            result += 1
        return result

class Solution3(object):

    def reinitializePermutation(self, n):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :rtype: int\n        '
        (result, i) = (0, 1)
        while not result or i != 1:
            i = i // 2 if not i % 2 else n // 2 + (i - 1) // 2
            result += 1
        return result