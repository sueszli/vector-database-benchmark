class Solution(object):

    def triangleNumber(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = 0
        nums.sort()
        for i in reversed(xrange(2, len(nums))):
            (left, right) = (0, i - 1)
            while left < right:
                if nums[left] + nums[right] > nums[i]:
                    result += right - left
                    right -= 1
                else:
                    left += 1
        return result

class Solution2(object):

    def triangleNumber(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = 0
        nums.sort()
        for i in xrange(len(nums) - 2):
            if nums[i] == 0:
                continue
            k = i + 2
            for j in xrange(i + 1, len(nums) - 1):
                while k < len(nums) and nums[i] + nums[j] > nums[k]:
                    k += 1
                result += k - j - 1
        return result