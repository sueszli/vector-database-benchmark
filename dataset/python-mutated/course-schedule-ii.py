import collections

class Solution(object):

    def findOrder(self, numCourses, prerequisites):
        if False:
            print('Hello World!')
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
        return result if len(result) == numCourses else []
import collections

class Solution2(object):

    def findOrder(self, numCourses, prerequisites):
        if False:
            return 10
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
        return result if len(result) == numCourses else []
import collections

class Solution3(object):

    def findOrder(self, numCourses, prerequisites):
        if False:
            while True:
                i = 10
        '\n        :type numCourses: int\n        :type prerequisites: List[List[int]]\n        :rtype: List[int]\n        '
        (WHITE, GRAY, BLACK) = range(3)

        def dfs(u):
            if False:
                print('Hello World!')
            if lookup[u] != WHITE:
                return lookup[u] == BLACK
            lookup[u] = GRAY
            if any((not dfs(v) for v in adj[u])):
                return False
            lookup[u] = BLACK
            result.append(u)
            return True
        result = []
        adj = collections.defaultdict(list)
        for (u, v) in prerequisites:
            adj[u].append(v)
        lookup = collections.defaultdict(lambda : WHITE)
        for u in xrange(numCourses):
            if not dfs(u):
                return []
        return result
import collections

class Solution4(object):

    def findOrder(self, numCourses, prerequisites):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type numCourses: int\n        :type prerequisites: List[List[int]]\n        :rtype: List[int]\n        '
        (WHITE, GRAY, BLACK) = range(3)

        def dfs(u):
            if False:
                while True:
                    i = 10
            if lookup[u] != WHITE:
                return lookup[u] == BLACK
            lookup[u] = GRAY
            if any((not dfs(v) for v in adj[u])):
                return False
            lookup[u] = BLACK
            result.append(u)
            return True
        result = []
        adj = collections.defaultdict(list)
        for (u, v) in prerequisites:
            adj[v].append(u)
        lookup = collections.defaultdict(lambda : WHITE)
        for u in xrange(numCourses):
            if not dfs(u):
                return []
        result.reverse()
        return result