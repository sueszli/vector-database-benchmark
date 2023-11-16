import collections

class Solution(object):

    def countPairs(self, nums, k):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def gcd(x, y):
            if False:
                while True:
                    i = 10
            while y:
                (x, y) = (y, x % y)
            return x
        idxs = collections.defaultdict(list)
        for (i, x) in enumerate(nums):
            idxs[x].append(i)
        result = 0
        for idx in idxs.itervalues():
            gcds = collections.Counter()
            for i in idx:
                gcd_i = gcd(i, k)
                result += sum((cnt for (gcd_j, cnt) in gcds.iteritems() if gcd_i * gcd_j % k == 0))
                gcds[gcd_i] += 1
        return result
import collections

class Solution2(object):

    def countPairs(self, nums, k):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def gcd(x, y):
            if False:
                for i in range(10):
                    print('nop')
            while y:
                (x, y) = (y, x % y)
            return x
        cnts = collections.defaultdict(collections.Counter)
        for (i, x) in enumerate(nums):
            cnts[x][gcd(i, k)] += 1
        result = 0
        for cnt in cnts.itervalues():
            for x in cnt.iterkeys():
                for y in cnt.iterkeys():
                    if x > y or x * y % k:
                        continue
                    result += cnt[x] * cnt[y] if x != y else cnt[x] * (cnt[x] - 1) // 2
        return result
import collections

class Solution3(object):

    def countPairs(self, nums, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        idxs = collections.defaultdict(list)
        for (i, x) in enumerate(nums):
            idxs[x].append(i)
        return sum((idx[i] * idx[j] % k == 0 for idx in idxs.itervalues() for i in xrange(len(idx)) for j in xrange(i + 1, len(idx))))