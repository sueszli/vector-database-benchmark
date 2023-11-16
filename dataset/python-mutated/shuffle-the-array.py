class Solution(object):

    def shuffle(self, nums, n):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type n: int\n        :rtype: List[int]\n        '

        def index(i):
            if False:
                while True:
                    i = 10
            return 2 * i if i < n else 2 * (i - n) + 1
        for i in xrange(len(nums)):
            j = i
            while nums[i] >= 0:
                j = index(j)
                (nums[i], nums[j]) = (nums[j], ~nums[i])
        for i in xrange(len(nums)):
            nums[i] = ~nums[i]
        return nums