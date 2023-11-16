import collections
from functools import partial

def bipartiteMatch(graph):
    if False:
        return 10
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
                i = 10
                return i + 15
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
                    for i in range(10):
                        print('nop')
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
                    return 10
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
                    print('Hello World!')
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

class Solution(object):

    def maxStudents(self, seats):
        if False:
            while True:
                i = 10
        '\n        :type seats: List[List[str]]\n        :rtype: int\n        '
        directions = [(-1, -1), (0, -1), (1, -1), (-1, 1), (0, 1), (1, 1)]
        (E, count) = (collections.defaultdict(list), 0)
        for i in xrange(len(seats)):
            for j in xrange(len(seats[0])):
                if seats[i][j] != '.':
                    continue
                count += 1
                if j % 2:
                    continue
                for (dx, dy) in directions:
                    (ni, nj) = (i + dx, j + dy)
                    if 0 <= ni < len(seats) and 0 <= nj < len(seats[0]) and (seats[ni][nj] == '.'):
                        E[i * len(seats[0]) + j].append(ni * len(seats[0]) + nj)
        return count - len(bipartiteMatch(E)[0])

class Solution2(object):

    def maxStudents(self, seats):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type seats: List[List[str]]\n        :rtype: int\n        '
        directions = [(-1, -1), (0, -1), (1, -1), (-1, 1), (0, 1), (1, 1)]

        def dfs(seats, e, lookup, matching):
            if False:
                print('Hello World!')
            (i, j) = e
            for (dx, dy) in directions:
                (ni, nj) = (i + dx, j + dy)
                if 0 <= ni < len(seats) and 0 <= nj < len(seats[0]) and (seats[ni][nj] == '.') and (not lookup[ni][nj]):
                    lookup[ni][nj] = True
                    if matching[ni][nj] == -1 or dfs(seats, matching[ni][nj], lookup, matching):
                        matching[ni][nj] = e
                        return True
            return False

        def Hungarian(seats):
            if False:
                while True:
                    i = 10
            result = 0
            matching = [[-1] * len(seats[0]) for _ in xrange(len(seats))]
            for i in xrange(len(seats)):
                for j in xrange(0, len(seats[0]), 2):
                    if seats[i][j] != '.':
                        continue
                    lookup = [[False] * len(seats[0]) for _ in xrange(len(seats))]
                    if dfs(seats, (i, j), lookup, matching):
                        result += 1
            return result
        count = 0
        for i in xrange(len(seats)):
            for j in xrange(len(seats[0])):
                if seats[i][j] == '.':
                    count += 1
        return count - Hungarian(seats)

class Solution3(object):

    def maxStudents(self, seats):
        if False:
            while True:
                i = 10
        '\n        :type seats: List[List[str]]\n        :rtype: int\n        '

        def popcount(n):
            if False:
                return 10
            result = 0
            while n:
                n &= n - 1
                result += 1
            return result
        dp = {0: 0}
        for row in seats:
            invalid_mask = sum((1 << c for (c, v) in enumerate(row) if v == '#'))
            new_dp = {}
            for (mask1, v1) in dp.iteritems():
                for mask2 in xrange(1 << len(seats[0])):
                    if mask2 & invalid_mask or mask2 & mask1 << 1 or mask2 & mask1 >> 1 or mask2 & mask2 << 1 or mask2 & mask2 >> 1:
                        continue
                    new_dp[mask2] = max(new_dp.get(mask2, 0), v1 + popcount(mask2))
            dp = new_dp
        return max(dp.itervalues()) if dp else 0