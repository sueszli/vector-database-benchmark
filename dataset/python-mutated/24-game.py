from operator import add, sub, mul, truediv
from fractions import Fraction

class Solution(object):

    def judgePoint24(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: bool\n        '
        if len(nums) == 1:
            return abs(nums[0] - 24) < 1e-06
        ops = [add, sub, mul, truediv]
        for i in xrange(len(nums)):
            for j in xrange(len(nums)):
                if i == j:
                    continue
                next_nums = [nums[k] for k in xrange(len(nums)) if i != k != j]
                for op in ops:
                    if (op is add or op is mul) and j > i or (op == truediv and nums[j] == 0):
                        continue
                    next_nums.append(op(nums[i], nums[j]))
                    if self.judgePoint24(next_nums):
                        return True
                    next_nums.pop()
        return False

class Solution2(object):

    def judgePoint24(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: bool\n        '

        def dfs(nums):
            if False:
                i = 10
                return i + 15
            if len(nums) == 1:
                return nums[0] == 24
            ops = [add, sub, mul, truediv]
            for i in xrange(len(nums)):
                for j in xrange(len(nums)):
                    if i == j:
                        continue
                    next_nums = [nums[k] for k in xrange(len(nums)) if i != k != j]
                    for op in ops:
                        if (op is add or op is mul) and j > i or (op == truediv and nums[j] == 0):
                            continue
                        next_nums.append(op(nums[i], nums[j]))
                        if dfs(next_nums):
                            return True
                        next_nums.pop()
            return False
        return dfs(map(Fraction, nums))