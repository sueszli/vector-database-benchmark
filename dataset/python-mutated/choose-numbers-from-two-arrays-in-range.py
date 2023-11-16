import collections
import itertools

class Solution(object):

    def countSubranges(self, nums1, nums2):
        if False:
            print('Hello World!')
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        result = 0
        dp = collections.Counter()
        for (x, y) in itertools.izip(nums1, nums2):
            new_dp = collections.Counter()
            new_dp[x] += 1
            new_dp[-y] += 1
            for (v, c) in dp.iteritems():
                new_dp[v + x] = (new_dp[v + x] + c) % MOD
                new_dp[v - y] = (new_dp[v - y] + c) % MOD
            dp = new_dp
            result = (result + dp[0]) % MOD
        return result