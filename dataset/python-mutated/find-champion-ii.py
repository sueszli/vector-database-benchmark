class Solution2(object):

    def findChampion(self, n, edges):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: int\n        '
        lookup = [False] * n
        for (u, v) in edges:
            lookup[v] = True
        result = -1
        for u in xrange(n):
            if lookup[u]:
                continue
            if result != -1:
                return -1
            result = u
        return result

class Solution2(object):

    def findChampion(self, n, edges):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: int\n        '
        lookup = {v for (_, v) in edges}
        return next((u for u in xrange(n) if u not in lookup)) if len(lookup) == n - 1 else -1