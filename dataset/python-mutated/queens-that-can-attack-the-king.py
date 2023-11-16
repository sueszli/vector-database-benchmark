class Solution(object):

    def queensAttacktheKing(self, queens, king):
        if False:
            i = 10
            return i + 15
        '\n        :type queens: List[List[int]]\n        :type king: List[int]\n        :rtype: List[List[int]]\n        '
        dirctions = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
        result = []
        lookup = {(i, j) for (i, j) in queens}
        for (dx, dy) in dirctions:
            for i in xrange(1, 8):
                (x, y) = (king[0] + dx * i, king[1] + dy * i)
                if (x, y) in lookup:
                    result.append([x, y])
                    break
        return result