import collections

class Solution(object):

    def minAreaRect(self, points):
        if False:
            return 10
        '\n        :type points: List[List[int]]\n        :rtype: int\n        '
        nx = len(set((x for (x, y) in points)))
        ny = len(set((y for (x, y) in points)))
        p = collections.defaultdict(list)
        if nx > ny:
            for (x, y) in points:
                p[x].append(y)
        else:
            for (x, y) in points:
                p[y].append(x)
        lookup = {}
        result = float('inf')
        for x in sorted(p):
            p[x].sort()
            for j in xrange(len(p[x])):
                for i in xrange(j):
                    (y1, y2) = (p[x][i], p[x][j])
                    if (y1, y2) in lookup:
                        result = min(result, (x - lookup[y1, y2]) * (y2 - y1))
                    lookup[y1, y2] = x
        return result if result != float('inf') else 0

class Solution2(object):

    def minAreaRect(self, points):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type points: List[List[int]]\n        :rtype: int\n        '
        lookup = set()
        result = float('inf')
        for (x1, y1) in points:
            for (x2, y2) in lookup:
                if (x1, y2) in lookup and (x2, y1) in lookup:
                    result = min(result, abs(x1 - x2) * abs(y1 - y2))
            lookup.add((x1, y1))
        return result if result != float('inf') else 0