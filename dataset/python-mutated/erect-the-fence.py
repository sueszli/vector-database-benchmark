class Solution(object):

    def outerTrees(self, points):
        if False:
            print('Hello World!')
        '\n        :type points: List[List[int]]\n        :rtype: List[List[int]]\n        '
        points = sorted(set((tuple(x) for x in points)))
        if len(points) <= 1:
            return points

        def cross(o, a, b):
            if False:
                for i in range(10):
                    print('nop')
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) < 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) < 0:
                upper.pop()
            upper.append(p)
        result = lower[:-1] + upper[:-1]
        return result if result[1] != result[-1] else result[:len(result) // 2 + 1]