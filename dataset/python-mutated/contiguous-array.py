class Solution(object):

    def findMaxLength(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        (result, count) = (0, 0)
        lookup = {0: -1}
        for (i, num) in enumerate(nums):
            count += 1 if num == 1 else -1
            if count in lookup:
                result = max(result, i - lookup[count])
            else:
                lookup[count] = i
        return result