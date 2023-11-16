class Solution(object):

    def nextGreaterElement(self, findNums, nums):
        if False:
            while True:
                i = 10
        '\n        :type findNums: List[int]\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        (stk, lookup) = ([], {})
        for num in nums:
            while stk and num > stk[-1]:
                lookup[stk.pop()] = num
            stk.append(num)
        while stk:
            lookup[stk.pop()] = -1
        return map(lambda x: lookup[x], findNums)