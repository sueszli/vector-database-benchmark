class Solution(object):

    def intersection(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[List[int]]\n        :rtype: List[int]\n        '
        MAX_NUM = 1000
        cnt = [0] * (MAX_NUM + 1)
        for num in nums:
            for x in num:
                cnt[x] += 1
        return [i for i in xrange(1, MAX_NUM + 1) if cnt[i] == len(nums)]

class Solution2(object):

    def intersection(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[List[int]]\n        :rtype: List[int]\n        '
        result = set(nums[0])
        for i in xrange(1, len(nums)):
            result = set((x for x in nums[i] if x in result))
        return [i for i in xrange(min(result), max(result) + 1) if i in result] if result else []

class Solution3(object):

    def intersection(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[List[int]]\n        :rtype: List[int]\n        '
        result = set(nums[0])
        for i in xrange(1, len(nums)):
            result = set((x for x in nums[i] if x in result))
        return sorted(result)