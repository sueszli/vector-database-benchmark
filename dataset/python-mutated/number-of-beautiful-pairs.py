class Solution(object):

    def countBeautifulPairs(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def gcd(a, b):
            if False:
                for i in range(10):
                    print('nop')
            while b:
                (a, b) = (b, a % b)
            return a
        result = 0
        cnt = [0] * 10
        for x in nums:
            for i in xrange(1, 10):
                if gcd(i, x % 10) == 1:
                    result += cnt[i]
            while x >= 10:
                x //= 10
            cnt[x] += 1
        return result