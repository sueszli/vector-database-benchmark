class Solution:

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        if False:
            i = 10
            return i + 15
        num_to_index = {}
        for (i, num) in enumerate(nums):
            if target - num in num_to_index:
                return [num_to_index[target - num], i]
            num_to_index[num] = i
        return []