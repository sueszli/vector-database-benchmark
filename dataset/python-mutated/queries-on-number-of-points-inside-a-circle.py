class Solution(object):

    def countPoints(self, points, queries):
        if False:
            return 10
        '\n        :type points: List[List[int]]\n        :type queries: List[List[int]]\n        :rtype: List[int]\n        '
        result = []
        for (i, j, r) in queries:
            result.append(0)
            for (x, y) in points:
                if (x - i) ** 2 + (y - j) ** 2 <= r ** 2:
                    result[-1] += 1
        return result