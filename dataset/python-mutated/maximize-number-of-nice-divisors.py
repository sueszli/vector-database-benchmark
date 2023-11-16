class Solution(object):

    def maxNiceDivisors(self, primeFactors):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type primeFactors: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        if primeFactors <= 3:
            return primeFactors
        if primeFactors % 3 == 0:
            return pow(3, primeFactors // 3, MOD)
        if primeFactors % 3 == 1:
            return 2 * 2 * pow(3, (primeFactors - 4) // 3, MOD) % MOD
        return 2 * pow(3, (primeFactors - 2) // 3, MOD) % MOD