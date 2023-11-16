class Solution(object):

    def minElements(self, nums, limit, goal):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type limit: int\n        :type goal: int\n        :rtype: int\n        '
        return (abs(sum(nums) - goal) + (limit - 1)) // limit