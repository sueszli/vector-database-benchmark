class Solution(object):

    def mostCompetitive(self, nums, k):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: List[int]\n        '
        stk = []
        for (i, x) in enumerate(nums):
            while stk and stk[-1] > x and (len(stk) + (len(nums) - i) > k):
                stk.pop()
            if len(stk) < k:
                stk.append(x)
        return stk