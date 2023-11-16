class Solution(object):

    def countGoodNumbers(self, n):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :rtype: int\n        '

        def powmod(a, b, mod):
            if False:
                print('Hello World!')
            a %= mod
            result = 1
            while b:
                if b & 1:
                    result = result * a % mod
                a = a * a % mod
                b >>= 1
            return result
        MOD = 10 ** 9 + 7
        return powmod(5, (n + 1) // 2 % (MOD - 1), MOD) * powmod(4, n // 2 % (MOD - 1), MOD) % MOD

class Solution2(object):

    def countGoodNumbers(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        return pow(5, (n + 1) // 2 % (MOD - 1), MOD) * pow(4, n // 2 % (MOD - 1), MOD) % MOD