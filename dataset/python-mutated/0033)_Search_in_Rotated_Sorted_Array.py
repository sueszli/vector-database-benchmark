class Solution:

    def search(self, nums: List[int], target: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        (left, right) = (0, len(nums) - 1)
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if nums[left] <= nums[mid]:
                if target >= nums[left] and target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            elif target <= nums[right] and target > nums[mid]:
                left = mid + 1
            else:
                right = mid - 1
        return -1