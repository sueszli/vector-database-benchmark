import itertools

class Solution(object):

    def makeSimilar(self, nums, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type target: List[int]\n        :rtype: int\n        '
        nums.sort(key=lambda x: (x % 2, x))
        target.sort(key=lambda x: (x % 2, x))
        return sum((abs(x - y) // 2 for (x, y) in itertools.izip(nums, target))) // 2