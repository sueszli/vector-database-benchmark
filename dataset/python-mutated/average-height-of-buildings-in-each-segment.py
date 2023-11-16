class Solution(object):

    def averageHeightOfBuildings(self, buildings):
        if False:
            print('Hello World!')
        '\n        :type buildings: List[List[int]]\n        :rtype: List[List[int]]\n        '
        points = []
        for (x, y, h) in buildings:
            points.append((x, 1, h))
            points.append((y, -1, h))
        points.sort()
        result = []
        total = cnt = 0
        prev = -1
        for (curr, c, h) in points:
            if cnt and curr != prev:
                if result and result[-1][1] == prev and (result[-1][2] == total // cnt):
                    result[-1][1] = curr
                else:
                    result.append([prev, curr, total // cnt])
            total += h * c
            cnt += c
            prev = curr
        return result
import collections

class Solution2(object):

    def averageHeightOfBuildings(self, buildings):
        if False:
            i = 10
            return i + 15
        '\n        :type buildings: List[List[int]]\n        :rtype: List[List[int]]\n        '
        count = collections.defaultdict(lambda : (0, 0))
        for (x, y, h) in buildings:
            count[x] = (count[x][0] + 1, count[x][1] + h)
            count[y] = (count[y][0] - 1, count[y][1] - h)
        result = []
        total = cnt = 0
        prev = -1
        for (curr, (c, h)) in sorted(count.iteritems()):
            if cnt:
                if result and result[-1][1] == prev and (result[-1][2] == total // cnt):
                    result[-1][1] = curr
                else:
                    result.append([prev, curr, total // cnt])
            total += h
            cnt += c
            prev = curr
        return result