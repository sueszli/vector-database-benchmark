class Solution(object):

    def minCost(self, startPos, homePos, rowCosts, colCosts):
        if False:
            print('Hello World!')
        '\n        :type startPos: List[int]\n        :type homePos: List[int]\n        :type rowCosts: List[int]\n        :type colCosts: List[int]\n        :rtype: int\n        '
        ([x0, y0], [x1, y1]) = (startPos, homePos)
        return sum((rowCosts[i] for i in xrange(min(x0, x1), max(x0, x1) + 1))) - rowCosts[x0] + (sum((colCosts[i] for i in xrange(min(y0, y1), max(y0, y1) + 1))) - colCosts[y0])