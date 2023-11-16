class Solution(object):

    def minDistance(self, height, width, tree, squirrel, nuts):
        if False:
            return 10
        '\n        :type height: int\n        :type width: int\n        :type tree: List[int]\n        :type squirrel: List[int]\n        :type nuts: List[List[int]]\n        :rtype: int\n        '

        def distance(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        result = 0
        d = float('inf')
        for nut in nuts:
            result += distance(nut, tree) * 2
            d = min(d, distance(nut, squirrel) - distance(nut, tree))
        return result + d