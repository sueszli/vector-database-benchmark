class Solution(object):

    def nearestValidPoint(self, x, y, points):
        if False:
            print('Hello World!')
        '\n        :type x: int\n        :type y: int\n        :type points: List[List[int]]\n        :rtype: int\n        '
        (smallest, idx) = (float('inf'), -1)
        for (i, (r, c)) in enumerate(points):
            (dx, dy) = (x - r, y - c)
            if dx * dy == 0 and abs(dx) + abs(dy) < smallest:
                smallest = abs(dx) + abs(dy)
                idx = i
        return idx