class Solution(object):

    def findSmallestSetOfVertices(self, n, edges):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: List[int]\n        '
        result = []
        lookup = set()
        for (u, v) in edges:
            lookup.add(v)
        for i in xrange(n):
            if i not in lookup:
                result.append(i)
        return result