import collections

class Solution(object):

    def validPath(self, n, edges, start, end):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type start: int\n        :type end: int\n        :rtype: bool\n        '

        def bi_bfs(adj, start, target):
            if False:
                return 10
            (left, right) = ({start}, {target})
            lookup = set()
            steps = 0
            while left:
                for pos in left:
                    lookup.add(pos)
                new_left = set()
                for pos in left:
                    if pos in right:
                        return steps
                    for nei in adj[pos]:
                        if nei in lookup:
                            continue
                        new_left.add(nei)
                left = new_left
                steps += 1
                if len(left) > len(right):
                    (left, right) = (right, left)
            return -1
        adj = collections.defaultdict(list)
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        return bi_bfs(adj, start, end) >= 0

class Solution2(object):

    def validPath(self, n, edges, start, end):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type start: int\n        :type end: int\n        :rtype: bool\n        '

        def bfs(adj, start, target):
            if False:
                for i in range(10):
                    print('nop')
            q = [start]
            lookup = set(q)
            steps = 0
            while q:
                new_q = []
                for pos in q:
                    if pos == target:
                        return steps
                    for nei in adj[pos]:
                        if nei in lookup:
                            continue
                        lookup.add(nei)
                        new_q.append(nei)
                q = new_q
                steps += 1
            return -1
        adj = collections.defaultdict(list)
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        return bfs(adj, start, end) >= 0

class Solution3(object):

    def validPath(self, n, edges, start, end):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type start: int\n        :type end: int\n        :rtype: bool\n        '

        def dfs(adj, start, target):
            if False:
                return 10
            stk = [start]
            lookup = set(stk)
            while stk:
                pos = stk.pop()
                if pos == target:
                    return True
                for nei in reversed(adj[pos]):
                    if nei in lookup:
                        continue
                    lookup.add(nei)
                    stk.append(nei)
            return False
        adj = collections.defaultdict(list)
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        return dfs(adj, start, end)