class Solution(object):

    def maximalRectangle(self, matrix):
        if False:
            return 10
        '\n        :type matrix: List[List[str]]\n        :rtype: int\n        '

        def largestRectangleArea(heights):
            if False:
                return 10
            (stk, result, i) = ([-1], 0, 0)
            for i in xrange(len(heights) + 1):
                while stk[-1] != -1 and (i == len(heights) or heights[stk[-1]] >= heights[i]):
                    result = max(result, heights[stk.pop()] * (i - 1 - stk[-1]))
                stk.append(i)
            return result
        if not matrix:
            return 0
        result = 0
        heights = [0] * len(matrix[0])
        for i in xrange(len(matrix)):
            for j in xrange(len(matrix[0])):
                heights[j] = heights[j] + 1 if matrix[i][j] == '1' else 0
            result = max(result, largestRectangleArea(heights))
        return result

class Solution2(object):

    def maximalRectangle(self, matrix):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type matrix: List[List[str]]\n        :rtype: int\n        '
        if not matrix:
            return 0
        result = 0
        m = len(matrix)
        n = len(matrix[0])
        L = [0 for _ in xrange(n)]
        H = [0 for _ in xrange(n)]
        R = [n for _ in xrange(n)]
        for i in xrange(m):
            left = 0
            for j in xrange(n):
                if matrix[i][j] == '1':
                    L[j] = max(L[j], left)
                    H[j] += 1
                else:
                    L[j] = 0
                    H[j] = 0
                    R[j] = n
                    left = j + 1
            right = n
            for j in reversed(xrange(n)):
                if matrix[i][j] == '1':
                    R[j] = min(R[j], right)
                    result = max(result, H[j] * (R[j] - L[j]))
                else:
                    right = j
        return result