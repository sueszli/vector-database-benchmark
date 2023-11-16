class Solution(object):

    def minMoves(self, nums, limit):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type limit: int\n        :rtype: int\n        '
        diff = [0] * (2 * (limit + 1))
        for i in xrange(len(nums) // 2):
            (left, right) = (nums[i], nums[-1 - i])
            diff[min(left, right) + 1] -= 1
            diff[left + right] -= 1
            diff[left + right + 1] += 1
            diff[max(left, right) + limit + 1] += 1
        result = count = len(nums)
        for total in xrange(2, 2 * limit + 1):
            count += diff[total]
            result = min(result, count)
        return result