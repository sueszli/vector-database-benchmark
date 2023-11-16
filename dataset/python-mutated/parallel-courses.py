import collections

class Solution(object):

    def minimumSemesters(self, N, relations):
        if False:
            while True:
                i = 10
        '\n        :type N: int\n        :type relations: List[List[int]]\n        :rtype: int\n        '
        g = collections.defaultdict(list)
        in_degree = [0] * N
        for (x, y) in relations:
            g[x - 1].append(y - 1)
            in_degree[y - 1] += 1
        q = collections.deque([(1, i) for i in xrange(N) if not in_degree[i]])
        result = 0
        count = N
        while q:
            (level, u) = q.popleft()
            count -= 1
            result = level
            for v in g[u]:
                in_degree[v] -= 1
                if not in_degree[v]:
                    q.append((level + 1, v))
        return result if count == 0 else -1