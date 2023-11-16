class Solution(object):

    def findBuildings(self, heights):
        if False:
            while True:
                i = 10
        '\n        :type heights: List[int]\n        :rtype: List[int]\n        '
        result = []
        for (i, h) in enumerate(heights):
            while result and heights[result[-1]] <= h:
                result.pop()
            result.append(i)
        return result

class Solution2(object):

    def findBuildings(self, heights):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type heights: List[int]\n        :rtype: List[int]\n        '
        result = []
        for i in reversed(xrange(len(heights))):
            if not result or heights[result[-1]] < heights[i]:
                result.append(i)
        result.reverse()
        return result