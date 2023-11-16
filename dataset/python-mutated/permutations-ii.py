class Solution(object):

    def permuteUnique(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: List[List[int]]\n        '
        nums.sort()
        result = []
        used = [False] * len(nums)
        self.permuteUniqueRecu(result, used, [], nums)
        return result

    def permuteUniqueRecu(self, result, used, cur, nums):
        if False:
            while True:
                i = 10
        if len(cur) == len(nums):
            result.append(cur + [])
            return
        for i in xrange(len(nums)):
            if used[i] or (i > 0 and nums[i - 1] == nums[i] and (not used[i - 1])):
                continue
            used[i] = True
            cur.append(nums[i])
            self.permuteUniqueRecu(result, used, cur, nums)
            cur.pop()
            used[i] = False

class Solution2(object):

    def permuteUnique(self, nums):
        if False:
            for i in range(10):
                print('nop')
        solutions = [[]]
        for num in nums:
            next = []
            for solution in solutions:
                for i in xrange(len(solution) + 1):
                    candidate = solution[:i] + [num] + solution[i:]
                    if candidate not in next:
                        next.append(candidate)
            solutions = next
        return solutions