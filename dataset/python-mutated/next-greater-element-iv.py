class Solution(object):

    def secondGreaterElement(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        (result, stk1, stk2) = ([-1] * len(nums), [], [])
        for (i, x) in enumerate(nums):
            while stk2 and nums[stk2[-1]] < x:
                result[stk2.pop()] = x
            tmp = []
            while stk1 and nums[stk1[-1]] < x:
                tmp.append(stk1.pop())
            stk1.append(i)
            for x in reversed(tmp):
                stk2.append(x)
        return result