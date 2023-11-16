import collections
import operator

class Solution(object):

    def countTheNumOfKFreeSubsets(self, nums, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def count(x):
            if False:
                print('Hello World!')
            y = x
            while y - k in cnt:
                y -= k
            dp = [1, 0]
            for i in xrange(y, x + 1, k):
                dp = [dp[0] + dp[1], dp[0] * ((1 << cnt[i]) - 1)]
            return sum(dp)
        cnt = collections.Counter(nums)
        return reduce(operator.mul, (count(i) for i in cnt.iterkeys() if i + k not in cnt))