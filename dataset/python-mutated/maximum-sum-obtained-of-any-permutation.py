import itertools

class Solution(object):

    def maxSumRangeQuery(self, nums, requests):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type requests: List[List[int]]\n        :rtype: int\n        '

        def addmod(a, b, mod):
            if False:
                for i in range(10):
                    print('nop')
            a %= mod
            b %= mod
            if mod - a <= b:
                b -= mod
            return a + b

        def mulmod(a, b, mod):
            if False:
                i = 10
                return i + 15
            a %= mod
            b %= mod
            if a < b:
                (a, b) = (b, a)
            result = 0
            while b > 0:
                if b % 2 == 1:
                    result = addmod(result, a, mod)
                a = addmod(a, a, mod)
                b //= 2
            return result
        MOD = 10 ** 9 + 7
        count = [0] * len(nums)
        for (start, end) in requests:
            count[start] += 1
            if end + 1 < len(count):
                count[end + 1] -= 1
        for i in xrange(1, len(count)):
            count[i] += count[i - 1]
        nums.sort()
        count.sort()
        result = 0
        for (i, (num, c)) in enumerate(itertools.izip(nums, count)):
            result = (result + num * c) % MOD
        return result