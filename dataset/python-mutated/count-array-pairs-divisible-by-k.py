import collections

class Solution(object):

    def countPairs(self, nums, k):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def gcd(x, y):
            if False:
                for i in range(10):
                    print('nop')
            while y:
                (x, y) = (y, x % y)
            return x
        cnt = collections.Counter()
        for x in nums:
            cnt[gcd(x, k)] += 1
        result = 0
        for x in cnt.iterkeys():
            for y in cnt.iterkeys():
                if x > y or x * y % k:
                    continue
                result += cnt[x] * cnt[y] if x != y else cnt[x] * (cnt[x] - 1) // 2
        return result
import collections

class Solution2(object):

    def countPairs(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def gcd(x, y):
            if False:
                while True:
                    i = 10
            while y:
                (x, y) = (y, x % y)
            return x
        result = 0
        gcds = collections.Counter()
        for x in nums:
            gcd_i = gcd(x, k)
            result += sum((cnt for (gcd_j, cnt) in gcds.iteritems() if gcd_i * gcd_j % k == 0))
            gcds[gcd_i] += 1
        return result