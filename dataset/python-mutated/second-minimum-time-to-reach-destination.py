class Solution(object):

    def secondMinimum(self, n, edges, time, change):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type time: int\n        :type change: int\n        :rtype: int\n        '

        def bi_bfs(adj, start, target):
            if False:
                i = 10
                return i + 15
            (left, right) = ({start}, {target})
            lookup = set()
            result = steps = 0
            while left and (not result or result + 2 > steps):
                for u in left:
                    lookup.add(u)
                new_left = set()
                for u in left:
                    if u in right:
                        if not result:
                            result = steps
                        elif result < steps:
                            return result + 1
                    for v in adj[u]:
                        if v in lookup:
                            continue
                        new_left.add(v)
                left = new_left
                steps += 1
                if len(left) > len(right):
                    (left, right) = (right, left)
            return result + 2

        def calc_time(time, change, dist):
            if False:
                i = 10
                return i + 15
            result = 0
            for _ in xrange(dist):
                if result // change % 2:
                    result = (result // change + 1) * change
                result += time
            return result
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            adj[u - 1].append(v - 1)
            adj[v - 1].append(u - 1)
        return calc_time(time, change, bi_bfs(adj, 0, n - 1))

class Solution2(object):

    def secondMinimum(self, n, edges, time, change):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type time: int\n        :type change: int\n        :rtype: int\n        '
        INF = float('inf')

        def bfs(adj, start):
            if False:
                while True:
                    i = 10
            q = [start]
            dist = [INF] * len(adj)
            dist[start] = 0
            while q:
                new_q = []
                for u in q:
                    for v in adj[u]:
                        if dist[v] != INF:
                            continue
                        dist[v] = dist[u] + 1
                        new_q.append(v)
                q = new_q
            return dist

        def calc_time(time, change, dist):
            if False:
                i = 10
                return i + 15
            result = 0
            for _ in xrange(dist):
                if result // change % 2:
                    result = (result // change + 1) * change
                result += time
            return result
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            adj[u - 1].append(v - 1)
            adj[v - 1].append(u - 1)
        (dist_to_end, dist_to_start) = (bfs(adj, 0), bfs(adj, n - 1))
        dist = dist_to_end[n - 1] + 2
        for i in xrange(n):
            if dist_to_end[i] + dist_to_start[i] == dist_to_end[n - 1]:
                continue
            dist = min(dist, dist_to_end[i] + dist_to_start[i])
            if dist == dist_to_end[n - 1] + 1:
                break
        return calc_time(time, change, dist)