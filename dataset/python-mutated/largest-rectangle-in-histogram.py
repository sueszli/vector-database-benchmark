class Solution(object):

    def largestRectangleArea(self, heights):
        if False:
            print('Hello World!')
        '\n        :type heights: List[int]\n        :rtype: int\n        '
        (stk, result) = ([-1], 0)
        for i in xrange(len(heights) + 1):
            while stk[-1] != -1 and (i == len(heights) or heights[stk[-1]] >= heights[i]):
                result = max(result, heights[stk.pop()] * (i - 1 - stk[-1]))
            stk.append(i)
        return result