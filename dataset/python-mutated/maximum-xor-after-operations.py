class Solution(object):

    def maximumXOR(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        return reduce(lambda x, y: x | y, nums)