import collections

class Solution(object):

    def possibleBipartition(self, N, dislikes):
        if False:
            print('Hello World!')
        '\n        :type N: int\n        :type dislikes: List[List[int]]\n        :rtype: bool\n        '
        adj = [[] for _ in xrange(N)]
        for (u, v) in dislikes:
            adj[u - 1].append(v - 1)
            adj[v - 1].append(u - 1)
        color = [0] * N
        color[0] = 1
        q = collections.deque([0])
        while q:
            cur = q.popleft()
            for nei in adj[cur]:
                if color[nei] == color[cur]:
                    return False
                elif color[nei] == -color[cur]:
                    continue
                color[nei] = -color[cur]
                q.append(nei)
        return True