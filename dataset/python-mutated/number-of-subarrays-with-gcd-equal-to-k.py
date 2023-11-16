class Solution(object):

    def subarrayGCD(self, nums, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def gcd(a, b):
            if False:
                for i in range(10):
                    print('nop')
            while b:
                (a, b) = (b, a % b)
            return a
        result = 0
        dp = collections.Counter()
        for x in nums:
            new_dp = collections.Counter()
            if x % k == 0:
                dp[x] += 1
                for (g, cnt) in dp.iteritems():
                    new_dp[gcd(g, x)] += cnt
            dp = new_dp
            result += dp[k]
        return result

class Solution2(object):

    def subarrayGCD(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def gcd(a, b):
            if False:
                return 10
            while b:
                (a, b) = (b, a % b)
            return a
        result = 0
        for i in xrange(len(nums)):
            g = 0
            for j in xrange(i, len(nums)):
                if nums[j] % k:
                    break
                g = gcd(g, nums[j])
                result += int(g == k)
        return result