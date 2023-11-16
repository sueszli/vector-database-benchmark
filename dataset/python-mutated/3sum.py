class Solution(object):

    def threeSum(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: List[List[int]]\n        '
        result = []
        nums.sort()
        for i in reversed(xrange(2, len(nums))):
            if i + 1 < len(nums) and nums[i] == nums[i + 1]:
                continue
            target = -nums[i]
            (left, right) = (0, i - 1)
            while left < right:
                if nums[left] + nums[right] < target:
                    left += 1
                elif nums[left] + nums[right] > target:
                    right -= 1
                else:
                    result.append([nums[left], nums[right], nums[i]])
                    left += 1
                    right -= 1
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1
        return result

class Solution2(object):

    def threeSum(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: List[List[int]]\n        '
        (nums, result, i) = (sorted(nums), [], 0)
        while i < len(nums) - 2:
            if i == 0 or nums[i] != nums[i - 1]:
                (j, k) = (i + 1, len(nums) - 1)
                while j < k:
                    if nums[i] + nums[j] + nums[k] < 0:
                        j += 1
                    elif nums[i] + nums[j] + nums[k] > 0:
                        k -= 1
                    else:
                        result.append([nums[i], nums[j], nums[k]])
                        (j, k) = (j + 1, k - 1)
                        while j < k and nums[j] == nums[j - 1]:
                            j += 1
                        while j < k and nums[k] == nums[k + 1]:
                            k -= 1
            i += 1
        return result