class Solution(object):

    def maxProduct(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        m1 = m2 = 0
        for num in nums:
            if num > m1:
                (m1, m2) = (num, m1)
            elif num > m2:
                m2 = num
        return (m1 - 1) * (m2 - 1)