import collections
import itertools

class Solution(object):

    def minAreaFreeRect(self, points):
        if False:
            print('Hello World!')
        '\n        :type points: List[List[int]]\n        :rtype: float\n        '
        points.sort()
        points = [complex(*z) for z in points]
        lookup = collections.defaultdict(list)
        for (P, Q) in itertools.combinations(points, 2):
            lookup[P - Q].append((P + Q) / 2)
        result = float('inf')
        for (A, candidates) in lookup.iteritems():
            for (P, Q) in itertools.combinations(candidates, 2):
                if A.real * (P - Q).real + A.imag * (P - Q).imag == 0.0:
                    result = min(result, abs(A) * abs(P - Q))
        return result if result < float('inf') else 0.0