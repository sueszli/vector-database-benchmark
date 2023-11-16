class Solution(object):

    def networkBecomesIdle(self, edges, patience):
        if False:
            return 10
        '\n        :type edges: List[List[int]]\n        :type patience: List[int]\n        :rtype: int\n        '
        adj = [[] for _ in xrange(len(patience))]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        q = [0]
        lookup = [False] * len(patience)
        lookup[0] = True
        step = 1
        result = 0
        while q:
            new_q = []
            for u in q:
                for v in adj[u]:
                    if lookup[v]:
                        continue
                    lookup[v] = True
                    new_q.append(v)
                    result = max(result, (step * 2 - 1) // patience[v] * patience[v] + step * 2)
            q = new_q
            step += 1
        return 1 + result