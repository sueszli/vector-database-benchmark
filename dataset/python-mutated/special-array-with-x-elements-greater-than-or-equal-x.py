class Solution(object):

    def specialArray(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        MAX_NUM = 1000
        count = [0] * (MAX_NUM + 1)
        for num in nums:
            count[num] += 1
        n = len(nums)
        for i in xrange(len(count)):
            if i == n:
                return i
            n -= count[i]
        return -1

class Solution2(object):

    def specialArray(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        MAX_NUM = 1000

        def inplace_counting_sort(nums, reverse=False):
            if False:
                i = 10
                return i + 15
            count = [0] * (MAX_NUM + 1)
            for num in nums:
                count[num] += 1
            for i in xrange(1, len(count)):
                count[i] += count[i - 1]
            for i in reversed(xrange(len(nums))):
                while nums[i] >= 0:
                    count[nums[i]] -= 1
                    j = count[nums[i]]
                    (nums[i], nums[j]) = (nums[j], ~nums[i])
            for i in xrange(len(nums)):
                nums[i] = ~nums[i]
            if reverse:
                nums.reverse()
        inplace_counting_sort(nums, reverse=True)
        (left, right) = (0, len(nums) - 1)
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] <= mid:
                right = mid - 1
            else:
                left = mid + 1
        return -1 if left < len(nums) and nums[left] == left else left

class Solution3(object):

    def specialArray(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        MAX_NUM = 1000

        def counting_sort(nums, reverse=False):
            if False:
                for i in range(10):
                    print('nop')
            count = [0] * (MAX_NUM + 1)
            for num in nums:
                count[num] += 1
            for i in xrange(1, len(count)):
                count[i] += count[i - 1]
            result = [0] * len(nums)
            if not reverse:
                for num in reversed(nums):
                    count[num] -= 1
                    result[count[num]] = num
            else:
                for num in nums:
                    count[num] -= 1
                    result[count[num]] = num
                result.reverse()
            return result
        nums = counting_sort(nums, reverse=True)
        (left, right) = (0, len(nums) - 1)
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] <= mid:
                right = mid - 1
            else:
                left = mid + 1
        return -1 if left < len(nums) and nums[left] == left else left

class Solution4(object):

    def specialArray(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        nums.sort(reverse=True)
        for i in xrange(len(nums)):
            if nums[i] <= i:
                break
        else:
            i += 1
        return -1 if i < len(nums) and nums[i] == i else i