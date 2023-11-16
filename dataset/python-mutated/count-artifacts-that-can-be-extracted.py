class Solution(object):

    def digArtifacts(self, n, artifacts, dig):
        if False:
            return 10
        '\n        :type n: int\n        :type artifacts: List[List[int]]\n        :type dig: List[List[int]]\n        :rtype: int\n        '
        lookup = set(map(tuple, dig))
        return sum((all(((i, j) in lookup for i in xrange(r1, r2 + 1) for j in xrange(c1, c2 + 1))) for (r1, c1, r2, c2) in artifacts))

class Solution2(object):

    def digArtifacts(self, n, artifacts, dig):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type artifacts: List[List[int]]\n        :type dig: List[List[int]]\n        :rtype: int\n        '
        lookup = {(i, j): idx for (idx, (r1, c1, r2, c2)) in enumerate(artifacts) for i in xrange(r1, r2 + 1) for j in xrange(c1, c2 + 1)}
        cnt = [(r2 - r1 + 1) * (c2 - c1 + 1) for (r1, c1, r2, c2) in artifacts]
        result = 0
        for (i, j) in dig:
            if (i, j) not in lookup:
                continue
            cnt[lookup[i, j]] -= 1
            if not cnt[lookup[i, j]]:
                result += 1
        return result