class Solution(object):

    def containsNearbyDuplicate(self, nums, k):
        if False:
            return 10
        lookup = {}
        for (i, num) in enumerate(nums):
            if num not in lookup:
                lookup[num] = i
            else:
                if i - lookup[num] <= k:
                    return True
                lookup[num] = i
        return False