class Interval(object):

    def __init__(self, s=0, e=0):
        if False:
            print('Hello World!')
        self.start = s
        self.end = e

class Solution(object):

    def intervalIntersection(self, A, B):
        if False:
            while True:
                i = 10
        '\n        :type A: List[Interval]\n        :type B: List[Interval]\n        :rtype: List[Interval]\n        '
        result = []
        (i, j) = (0, 0)
        while i < len(A) and j < len(B):
            left = max(A[i].start, B[j].start)
            right = min(A[i].end, B[j].end)
            if left <= right:
                result.append(Interval(left, right))
            if A[i].end < B[j].end:
                i += 1
            else:
                j += 1
        return result