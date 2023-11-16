import bisect

class Solution(object):

    def answerQueries(self, nums, queries):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type queries: List[int]\n        :rtype: List[int]\n        '
        nums.sort()
        for i in xrange(len(nums) - 1):
            nums[i + 1] += nums[i]
        return [bisect.bisect_right(nums, q) for q in queries]