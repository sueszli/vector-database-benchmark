from collections import defaultdict

class Solution(object):

    def isRectangleCover(self, rectangles):
        if False:
            while True:
                i = 10
        '\n        :type rectangles: List[List[int]]\n        :rtype: bool\n        '
        left = min((rec[0] for rec in rectangles))
        bottom = min((rec[1] for rec in rectangles))
        right = max((rec[2] for rec in rectangles))
        top = max((rec[3] for rec in rectangles))
        points = defaultdict(int)
        for (l, b, r, t) in rectangles:
            for (p, q) in zip(((l, b), (r, b), (l, t), (r, t)), (1, 2, 4, 8)):
                if points[p] & q:
                    return False
                points[p] |= q
        for (px, py) in points:
            if left < px < right or bottom < py < top:
                if points[px, py] not in (3, 5, 10, 12, 15):
                    return False
        return True