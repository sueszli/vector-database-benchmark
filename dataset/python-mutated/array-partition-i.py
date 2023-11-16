class Solution(object):

    def arrayPairSum(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        (LEFT, RIGHT) = (-10000, 10000)
        lookup = [0] * (RIGHT - LEFT + 1)
        for num in nums:
            lookup[num - LEFT] += 1
        (r, result) = (0, 0)
        for i in xrange(LEFT, RIGHT + 1):
            result += (lookup[i - LEFT] + 1 - r) / 2 * i
            r = (lookup[i - LEFT] + r) % 2
        return result

class Solution2(object):

    def arrayPairSum(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        nums.sort()
        result = 0
        for i in xrange(0, len(nums), 2):
            result += nums[i]
        return result

class Solution3(object):

    def arrayPairSum(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        nums = sorted(nums)
        return sum([nums[i] for i in range(0, len(nums), 2)])