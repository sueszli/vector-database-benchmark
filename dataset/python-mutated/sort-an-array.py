class Solution(object):

    def sortArray(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '

        def mergeSort(left, right, nums):
            if False:
                print('Hello World!')
            if left == right:
                return
            mid = left + (right - left) // 2
            mergeSort(left, mid, nums)
            mergeSort(mid + 1, right, nums)
            r = mid + 1
            tmp = []
            for l in xrange(left, mid + 1):
                while r <= right and nums[r] < nums[l]:
                    tmp.append(nums[r])
                    r += 1
                tmp.append(nums[l])
            nums[left:left + len(tmp)] = tmp
        mergeSort(0, len(nums) - 1, nums)
        return nums
import random

class Solution2(object):

    def sortArray(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '

        def nth_element(nums, left, n, right, compare=lambda a, b: a < b):
            if False:
                for i in range(10):
                    print('nop')

            def tri_partition(nums, left, right, target):
                if False:
                    while True:
                        i = 10
                i = left
                while i <= right:
                    if compare(nums[i], target):
                        (nums[i], nums[left]) = (nums[left], nums[i])
                        left += 1
                        i += 1
                    elif compare(target, nums[i]):
                        (nums[i], nums[right]) = (nums[right], nums[i])
                        right -= 1
                    else:
                        i += 1
                return (left, right)
            while left <= right:
                pivot_idx = random.randint(left, right)
                (pivot_left, pivot_right) = tri_partition(nums, left, right, nums[pivot_idx])
                if pivot_left <= n <= pivot_right:
                    return
                elif pivot_left > n:
                    right = pivot_left - 1
                else:
                    left = pivot_right + 1

        def quickSort(left, right, nums):
            if False:
                for i in range(10):
                    print('nop')
            if left > right:
                return
            mid = left + (right - left) // 2
            nth_element(nums, left, mid, right)
            quickSort(left, mid - 1, nums)
            quickSort(mid + 1, right, nums)
        quickSort(0, len(nums) - 1, nums)
        return nums