class Solution(object):

    def sumDistance(self, nums, s, d):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type s: str\n        :type d: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        for i in xrange(len(nums)):
            nums[i] += d if s[i] == 'R' else -d
        nums.sort()
        return reduce(lambda x, y: (x + y) % MOD, ((i - (len(nums) - (i + 1))) * x for (i, x) in enumerate(nums)))