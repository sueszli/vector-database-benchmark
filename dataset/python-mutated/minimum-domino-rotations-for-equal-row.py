import itertools

class Solution(object):

    def minDominoRotations(self, A, B):
        if False:
            return 10
        '\n        :type A: List[int]\n        :type B: List[int]\n        :rtype: int\n        '
        intersect = reduce(set.__and__, [set(d) for d in itertools.izip(A, B)])
        if not intersect:
            return -1
        x = intersect.pop()
        return min(len(A) - A.count(x), len(B) - B.count(x))