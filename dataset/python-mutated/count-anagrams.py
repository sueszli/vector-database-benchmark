import collections

class Solution(object):

    def countAnagrams(self, s):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        (fact, inv, inv_fact) = [[1] * 2 for _ in xrange(3)]

        def lazy_init(n):
            if False:
                for i in range(10):
                    print('nop')
            while len(inv) <= n:
                fact.append(fact[-1] * len(inv) % MOD)
                inv.append(inv[MOD % len(inv)] * (MOD - MOD // len(inv)) % MOD)
                inv_fact.append(inv_fact[-1] * inv[-1] % MOD)

        def factorial(n):
            if False:
                i = 10
                return i + 15
            lazy_init(n)
            return fact[n]

        def inv_factorial(n):
            if False:
                for i in range(10):
                    print('nop')
            lazy_init(n)
            return inv_fact[n]

        def count(j, i):
            if False:
                i = 10
                return i + 15
            result = 1
            cnt = collections.Counter()
            for k in xrange(j, i + 1):
                cnt[s[k]] += 1
            result = factorial(sum(cnt.itervalues()))
            for c in cnt.itervalues():
                result = result * inv_factorial(c) % MOD
            return result
        result = 1
        j = 0
        for i in xrange(len(s)):
            if i + 1 != len(s) and s[i + 1] != ' ':
                continue
            result = result * count(j, i) % MOD
            j = i + 2
        return result