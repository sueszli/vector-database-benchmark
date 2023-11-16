class Solution(object):

    def canPartitionKSubsets(self, nums, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: bool\n        '

        def dfs(nums, target, used, todo, lookup):
            if False:
                while True:
                    i = 10
            if lookup[used] is None:
                targ = (todo - 1) % target + 1
                lookup[used] = any((dfs(nums, target, used | 1 << i, todo - num, lookup) for (i, num) in enumerate(nums) if used >> i & 1 == 0 and num <= targ))
            return lookup[used]
        total = sum(nums)
        if total % k or max(nums) > total // k:
            return False
        lookup = [None] * (1 << len(nums))
        lookup[-1] = True
        return dfs(nums, total // k, 0, total, lookup)

class Solution2(object):

    def canPartitionKSubsets(self, nums, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: bool\n        '

        def dfs(nums, target, i, subset_sums):
            if False:
                while True:
                    i = 10
            if i == len(nums):
                return True
            for k in xrange(len(subset_sums)):
                if subset_sums[k] + nums[i] > target:
                    continue
                subset_sums[k] += nums[i]
                if dfs(nums, target, i + 1, subset_sums):
                    return True
                subset_sums[k] -= nums[i]
                if not subset_sums[k]:
                    break
            return False
        total = sum(nums)
        if total % k != 0 or max(nums) > total // k:
            return False
        nums.sort(reverse=True)
        subset_sums = [0] * k
        return dfs(nums, total // k, 0, subset_sums)