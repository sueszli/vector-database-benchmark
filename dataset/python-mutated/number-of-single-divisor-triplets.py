import collections
import itertools

class Solution(object):

    def singleDivisorTriplet(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def check(a, b, c):
            if False:
                for i in range(10):
                    print('nop')
            return sum(((a + b + c) % x == 0 for x in (a, b, c))) == 1
        cnt = collections.Counter(nums)
        return 6 * (sum((cnt[a] * cnt[b] * cnt[c] for (a, b, c) in itertools.combinations(cnt.keys(), 3) if check(a, b, c))) + sum((cnt[a] * (cnt[a] - 1) // 2 * cnt[b] for (a, b) in itertools.permutations(cnt.keys(), 2) if check(a, a, b))))