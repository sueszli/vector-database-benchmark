import collections

class Solution(object):

    def canFinish(self, numCourses, prerequisites):
        if False:
            i = 10
            return i + 15
        '\n        :type numCourses: int\n        :type prerequisites: List[List[int]]\n        :rtype: List[int]\n        '
        adj = collections.defaultdict(list)
        in_degree = collections.Counter()
        for (u, v) in prerequisites:
            in_degree[u] += 1
            adj[v].append(u)
        result = []
        q = [u for u in xrange(numCourses) if u not in in_degree]
        while q:
            new_q = []
            for u in q:
                result.append(u)
                for v in adj[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        new_q.append(v)
            q = new_q
        return len(result) == numCourses
import collections

class Solution2(object):

    def canFinish(self, numCourses, prerequisites):
        if False:
            i = 10
            return i + 15
        '\n        :type numCourses: int\n        :type prerequisites: List[List[int]]\n        :rtype: List[int]\n        '
        adj = collections.defaultdict(list)
        in_degree = collections.Counter()
        for (u, v) in prerequisites:
            in_degree[u] += 1
            adj[v].append(u)
        result = []
        stk = [u for u in xrange(numCourses) if u not in in_degree]
        while stk:
            u = stk.pop()
            result.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    stk.append(v)
        return len(result) == numCourses