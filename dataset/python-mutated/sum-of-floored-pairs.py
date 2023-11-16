import collections

class Solution(object):

    def sumOfFlooredPairs(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        (prefix, counter) = ([0] * (max(nums) + 1), collections.Counter(nums))
        for (num, cnt) in counter.iteritems():
            for j in xrange(num, len(prefix), num):
                prefix[j] += counter[num]
        for i in xrange(len(prefix) - 1):
            prefix[i + 1] += prefix[i]
        return reduce(lambda total, num: (total + prefix[num]) % MOD, nums, 0)