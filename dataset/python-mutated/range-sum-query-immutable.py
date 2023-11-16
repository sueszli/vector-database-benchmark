class NumArray(object):

    def __init__(self, nums):
        if False:
            while True:
                i = 10
        '\n        initialize your data structure here.\n        :type nums: List[int]\n        '
        self.accu = [0]
        for num in nums:
            (self.accu.append(self.accu[-1] + num),)

    def sumRange(self, i, j):
        if False:
            for i in range(10):
                print('nop')
        '\n        sum of elements nums[i..j], inclusive.\n        :type i: int\n        :type j: int\n        :rtype: int\n        '
        return self.accu[j + 1] - self.accu[i]