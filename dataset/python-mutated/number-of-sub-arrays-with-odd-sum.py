class Solution(object):

    def numOfSubarrays(self, arr):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type arr: List[int]\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        (result, accu) = (0, 0)
        dp = [1, 0]
        for x in arr:
            accu ^= x & 1
            dp[accu] += 1
            result = (result + dp[accu ^ 1]) % MOD
        return result