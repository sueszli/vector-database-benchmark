import itertools
from fractions import gcd

class Solution(object):

    def maxScore(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def popcount(n):
            if False:
                print('Hello World!')
            count = 0
            while n:
                n &= n - 1
                count += 1
            return count

        def bits(mask):
            if False:
                return 10
            result = []
            i = 0
            while mask:
                if mask & 1:
                    result.append(i)
                i += 1
                mask >>= 1
            return result
        dp = [0] * 2 ** len(nums)
        for mask in xrange(3, len(dp)):
            cnt = popcount(mask)
            if cnt % 2:
                continue
            for (i, j) in itertools.combinations(bits(mask), 2):
                dp[mask] = max(dp[mask], cnt // 2 * gcd(nums[i], nums[j]) + dp[mask ^ 1 << i ^ 1 << j])
        return dp[-1]