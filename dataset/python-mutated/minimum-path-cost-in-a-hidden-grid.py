class GridMaster(object):

    def canMove(self, direction):
        if False:
            i = 10
            return i + 15
        pass

    def move(self, direction):
        if False:
            for i in range(10):
                print('nop')
        pass

    def isTarget(self):
        if False:
            return 10
        pass
import collections
import heapq

class Solution(object):

    def findShortestPath(self, master):
        if False:
            return 10
        '\n        :type master: GridMaster\n        :rtype: int\n        '
        directions = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
        rollback = {'L': 'R', 'R': 'L', 'U': 'D', 'D': 'U'}

        def dfs(pos, target, master, lookup, adj):
            if False:
                for i in range(10):
                    print('nop')
            if target[0] is None and master.isTarget():
                target[0] = pos
            lookup.add(pos)
            for (d, (di, dj)) in directions.iteritems():
                if not master.canMove(d):
                    continue
                nei = (pos[0] + di, pos[1] + dj)
                if nei in adj[pos]:
                    continue
                adj[pos][nei] = master.move(d)
                if nei not in lookup:
                    dfs(nei, target, master, lookup, adj)
                adj[nei][pos] = master.move(rollback[d])

        def dijkstra(adj, start, target):
            if False:
                i = 10
                return i + 15
            dist = {start: 0}
            min_heap = [(0, start)]
            while min_heap:
                (curr, u) = heapq.heappop(min_heap)
                if dist[u] < curr:
                    continue
                for (v, w) in adj[u].iteritems():
                    if v in dist and dist[v] <= curr + w:
                        continue
                    dist[v] = curr + w
                    heapq.heappush(min_heap, (curr + w, v))
            return dist[target] if target in dist else -1
        start = (0, 0)
        target = [None]
        adj = collections.defaultdict(dict)
        dfs(start, target, master, set(), adj)
        if not target[0]:
            return -1
        return dijkstra(adj, start, target[0])