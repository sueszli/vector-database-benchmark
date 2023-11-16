from functools import partial

def bipartiteMatch(graph):
    if False:
        i = 10
        return i + 15
    'Find maximum cardinality matching of a bipartite graph (U,V,E).\n    The input format is a dictionary mapping members of U to a list\n    of their neighbors in V.  The output is a triple (M,A,B) where M is a\n    dictionary mapping members of V to their matches in U, A is the part\n    of the maximum independent set in U, and B is the part of the MIS in V.\n    The same object may occur in both U and V, and is treated as two\n    distinct vertices if this happens.'
    matching = {}
    for u in graph:
        for v in graph[u]:
            if v not in matching:
                matching[v] = u
                break
    while 1:
        preds = {}
        unmatched = []
        pred = dict([(u, unmatched) for u in graph])
        for v in matching:
            del pred[matching[v]]
        layer = list(pred)
        while layer and (not unmatched):
            newLayer = {}
            for u in layer:
                for v in graph[u]:
                    if v not in preds:
                        newLayer.setdefault(v, []).append(u)
            layer = []
            for v in newLayer:
                preds[v] = newLayer[v]
                if v in matching:
                    layer.append(matching[v])
                    pred[matching[v]] = v
                else:
                    unmatched.append(v)
        if not unmatched:
            unlayered = {}
            for u in graph:
                for v in graph[u]:
                    if v not in preds:
                        unlayered[v] = None
            return (matching, list(pred), list(unlayered))

        def recurse(v):
            if False:
                return 10
            if v in preds:
                L = preds[v]
                del preds[v]
                for u in L:
                    if u in pred:
                        pu = pred[u]
                        del pred[u]
                        if pu is unmatched or recurse(pu):
                            matching[v] = u
                            return 1
            return 0

        def recurse_iter(v):
            if False:
                while True:
                    i = 10

            def divide(v):
                if False:
                    i = 10
                    return i + 15
                if v not in preds:
                    return
                L = preds[v]
                del preds[v]
                for u in L:
                    if u in pred and pred[u] is unmatched:
                        del pred[u]
                        matching[v] = u
                        ret[0] = True
                        return
                stk.append(partial(conquer, v, iter(L)))

            def conquer(v, it):
                if False:
                    while True:
                        i = 10
                for u in it:
                    if u not in pred:
                        continue
                    pu = pred[u]
                    del pred[u]
                    stk.append(partial(postprocess, v, u, it))
                    stk.append(partial(divide, pu))
                    return

            def postprocess(v, u, it):
                if False:
                    while True:
                        i = 10
                if not ret[0]:
                    stk.append(partial(conquer, v, it))
                    return
                matching[v] = u
            (ret, stk) = ([False], [])
            stk.append(partial(divide, v))
            while stk:
                stk.pop()()
            return ret[0]
        for v in unmatched:
            recurse_iter(v)
import collections

class Solution(object):

    def minimumOperations(self, grid):
        if False:
            print('Hello World!')
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def iter_dfs(grid, i, j, lookup, adj):
            if False:
                for i in range(10):
                    print('nop')
            if lookup[i][j]:
                return
            lookup[i][j] = True
            stk = [(i, j, (i + j) % 2)]
            while stk:
                (i, j, color) = stk.pop()
                for (di, dj) in directions:
                    (ni, nj) = (i + di, j + dj)
                    if not (0 <= ni < len(grid) and 0 <= nj < len(grid[0]) and grid[ni][nj]):
                        continue
                    if not color:
                        adj[len(grid[0]) * ni + nj].append(len(grid[0]) * i + j)
                    if lookup[ni][nj]:
                        continue
                    lookup[ni][nj] = True
                    stk.append((ni, nj, color ^ 1))
        adj = collections.defaultdict(list)
        lookup = [[False] * len(grid[0]) for _ in xrange(len(grid))]
        for i in xrange(len(grid)):
            for j in xrange(len(grid[0])):
                if not grid[i][j]:
                    continue
                iter_dfs(grid, i, j, lookup, adj)
        return len(bipartiteMatch(adj)[0])