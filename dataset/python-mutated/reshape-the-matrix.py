class Solution(object):

    def matrixReshape(self, nums, r, c):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[List[int]]\n        :type r: int\n        :type c: int\n        :rtype: List[List[int]]\n        '
        if not nums or r * c != len(nums) * len(nums[0]):
            return nums
        result = [[0 for _ in xrange(c)] for _ in xrange(r)]
        count = 0
        for i in xrange(len(nums)):
            for j in xrange(len(nums[0])):
                result[count / c][count % c] = nums[i][j]
                count += 1
        return result