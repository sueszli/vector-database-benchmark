class Solution(object):

    def arrayChange(self, nums, operations):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type operations: List[List[int]]\n        :rtype: List[int]\n        '
        lookup = {x: i for (i, x) in enumerate(nums)}
        for (x, y) in operations:
            lookup[y] = lookup.pop(x)
        for (x, i) in lookup.iteritems():
            nums[i] = x
        return nums

class Solution2(object):

    def arrayChange(self, nums, operations):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type operations: List[List[int]]\n        :rtype: List[int]\n        '
        lookup = {x: i for (i, x) in enumerate(nums)}
        for (x, y) in operations:
            nums[lookup[x]] = y
            lookup[y] = lookup.pop(x)
        return nums