import collections

class Solution(object):

    def subarrayLCM(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def gcd(a, b):
            if False:
                while True:
                    i = 10
            while b:
                (a, b) = (b, a % b)
            return a

        def lcm(a, b):
            if False:
                print('Hello World!')
            return a // gcd(a, b) * b
        result = 0
        dp = collections.Counter()
        for x in nums:
            new_dp = collections.Counter()
            if k % x == 0:
                dp[x] += 1
                for (l, cnt) in dp.iteritems():
                    new_dp[lcm(l, x)] += cnt
            dp = new_dp
            result += dp[k]
        return result

class Solution2(object):

    def subarrayLCM(self, nums, k):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def gcd(a, b):
            if False:
                print('Hello World!')
            while b:
                (a, b) = (b, a % b)
            return a

        def lcm(a, b):
            if False:
                print('Hello World!')
            return a // gcd(a, b) * b
        result = 0
        for i in xrange(len(nums)):
            l = 1
            for j in xrange(i, len(nums)):
                if k % nums[j]:
                    break
                l = lcm(l, nums[j])
                result += int(l == k)
        return result