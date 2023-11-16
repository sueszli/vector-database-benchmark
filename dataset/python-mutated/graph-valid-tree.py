import collections

class Solution(object):

    def validTree(self, n, edges):
        if False:
            for i in range(10):
                print('nop')
        if len(edges) != n - 1:
            return False
        neighbors = collections.defaultdict(list)
        for (u, v) in edges:
            neighbors[u].append(v)
            neighbors[v].append(u)
        q = collections.deque([0])
        visited = set([0])
        while q:
            curr = q.popleft()
            for node in neighbors[curr]:
                if node not in visited:
                    visited.add(node)
                    q.append(node)
        return len(visited) == n

class Solution2(object):

    def validTree(self, n, edges):
        if False:
            for i in range(10):
                print('nop')
        visited_from = [-1] * n
        neighbors = collections.defaultdict(list)
        for (u, v) in edges:
            neighbors[u].append(v)
            neighbors[v].append(u)
        q = collections.deque([0])
        visited = set([0])
        while q:
            i = q.popleft()
            for node in neighbors[i]:
                if node != visited_from[i]:
                    if node in visited:
                        return False
                    else:
                        visited.add(node)
                        visited_from[node] = i
                        q.append(node)
        return len(visited) == n