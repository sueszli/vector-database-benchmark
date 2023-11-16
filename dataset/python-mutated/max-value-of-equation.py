import collections

class Solution(object):

    def findMaxValueOfEquation(self, points, k):
        if False:
            while True:
                i = 10
        '\n        :type points: List[List[int]]\n        :type k: int\n        :rtype: int\n        '
        result = float('-inf')
        dq = collections.deque()
        for (i, (x, y)) in enumerate(points):
            while dq and points[dq[0]][0] < x - k:
                dq.popleft()
            if dq:
                result = max(result, points[dq[0]][1] - points[dq[0]][0] + y + x)
            while dq and points[dq[-1]][1] - points[dq[-1]][0] <= y - x:
                dq.pop()
            dq.append(i)
        return result