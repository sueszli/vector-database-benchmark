class Solution:

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        if False:
            return 10
        res = []
        nums.sort()
        self.dfs(nums, [], res)
        return res

    def dfs(self, nums, path, res):
        if False:
            i = 10
            return i + 15
        if not nums:
            res.append(path)
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            self.dfs(nums[:i] + nums[i + 1:], path + [nums[i]], res)