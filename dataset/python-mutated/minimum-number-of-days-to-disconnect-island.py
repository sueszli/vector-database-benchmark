def iter_get_articulation_points(graph, v):
    if False:
        print('Hello World!')

    def iter_dfs(v, p):
        if False:
            return 10
        stk = [(1, (v, p))]
        while stk:
            (step, args) = stk.pop()
            if step == 1:
                (v, p) = args
                index[v] = index_counter[0]
                lowlinks[v] = index_counter[0]
                index_counter[0] += 1
                children_count = [0]
                is_cut = [False]
                stk.append((4, (v, p, children_count, is_cut)))
                for w in reversed(graph[v]):
                    if w == p:
                        continue
                    stk.append((2, (w, v, children_count, is_cut)))
            elif step == 2:
                (w, v, children_count, is_cut) = args
                if index[w] == -1:
                    children_count[0] += 1
                    stk.append((3, (w, v, is_cut)))
                    stk.append((1, (w, v)))
                else:
                    lowlinks[v] = min(lowlinks[v], index[w])
            elif step == 3:
                (w, v, is_cut) = args
                if lowlinks[w] >= index[v]:
                    is_cut[0] = True
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif step == 4:
                (v, p, children_count, is_cut) = args
                if p != -1 and is_cut[0] or (p == -1 and children_count[0] >= 2):
                    cutpoints.append(v)
    (index_counter, index, lowlinks) = ([0], [-1] * len(graph), [0] * len(graph))
    cutpoints = []
    iter_dfs(v, -1)
    return cutpoints

class Solution(object):

    def minDays(self, grid):
        if False:
            i = 10
            return i + 15
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '
        DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def floodfill(grid, i, j, lookup):
            if False:
                i = 10
                return i + 15
            stk = [(i, j)]
            lookup[i][j] = 1
            while stk:
                (i, j) = stk.pop()
                for (di, dj) in DIRECTIONS:
                    (ni, nj) = (i + di, j + dj)
                    if not (0 <= ni < R and 0 <= nj < C and grid[ni][nj] and (not lookup[ni][nj])):
                        continue
                    lookup[ni][nj] = 1
                    stk.append((ni, nj))

        def count_islands(grid):
            if False:
                print('Hello World!')
            lookup = [[0] * C for _ in xrange(R)]
            island_cnt = 0
            for i in xrange(R):
                for j in xrange(C):
                    if grid[i][j] == 0 or lookup[i][j]:
                        continue
                    island_cnt += 1
                    floodfill(grid, i, j, lookup)
            return island_cnt
        (R, C) = (len(grid), len(grid[0]))
        if count_islands(grid) != 1:
            return 0
        adj = [[] for _ in xrange(R * C)]
        (cnt, idx) = (0, -1)
        for i in xrange(R):
            for j in xrange(C):
                if grid[i][j] == 0:
                    continue
                cnt += 1
                if idx == -1:
                    idx = i * C + j
                for (di, dj) in DIRECTIONS:
                    (ni, nj) = (i + di, j + dj)
                    if 0 <= ni < R and 0 <= nj < C and (grid[ni][nj] == grid[i][j]):
                        adj[i * C + j].append(ni * C + nj)
        return 1 if cnt == 1 or iter_get_articulation_points(adj, idx) else 2

class PersistentUnionFind(object):

    def __init__(self, n):
        if False:
            while True:
                i = 10
        self.set = range(n)
        self.size = [1] * n
        self.snapshots = []
        self.undos = []

    def find_set(self, x):
        if False:
            while True:
                i = 10
        stk = []
        while self.set[x] != x:
            stk.append(x)
            x = self.set[x]
        while stk:
            y = stk.pop()
            self.undos.append((~y, self.set[y]))
            self.set[y] = x
        return x

    def union_set(self, x, y):
        if False:
            while True:
                i = 10
        (x, y) = (self.find_set(x), self.find_set(y))
        if x == y:
            return False
        if self.size[x] > self.size[y]:
            (x, y) = (y, x)
        self.undos.append((x, y))
        self.set[x] = self.set[y]
        self.size[y] += self.size[x]
        return True

    def total(self, x):
        if False:
            return 10
        return self.size[self.find_set(x)]

    def snapshot(self):
        if False:
            i = 10
            return i + 15
        self.snapshots.append(len(self.undos))

    def rollback(self):
        if False:
            for i in range(10):
                print('nop')
        for _ in xrange(len(self.undos) - self.snapshots.pop()):
            (x, y) = self.undos.pop()
            if x >= 0:
                self.size[y] -= self.size[x]
                self.set[x] = x
            else:
                self.set[~x] = y

class Solution2(object):

    def minDays(self, grid):
        if False:
            i = 10
            return i + 15
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '
        DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def floodfill(grid, i, j, lookup):
            if False:
                i = 10
                return i + 15
            stk = [(i, j)]
            lookup[i][j] = 1
            while stk:
                (i, j) = stk.pop()
                for (di, dj) in DIRECTIONS:
                    (ni, nj) = (i + di, j + dj)
                    if not (0 <= ni < R and 0 <= nj < C and grid[ni][nj] and (not lookup[ni][nj])):
                        continue
                    lookup[ni][nj] = 1
                    stk.append((ni, nj))

        def count_islands(grid):
            if False:
                for i in range(10):
                    print('nop')
            lookup = [[0] * C for _ in xrange(R)]
            island_cnt = 0
            for i in xrange(R):
                for j in xrange(C):
                    if grid[i][j] == 0 or lookup[i][j]:
                        continue
                    island_cnt += 1
                    floodfill(grid, i, j, lookup)
            return island_cnt

        def merge(i):
            if False:
                while True:
                    i = 10
            (r, c) = divmod(i, C)
            for (dr, dc) in DIRECTIONS:
                (nr, nc) = (r + dr, c + dc)
                ni = nr * C + nc
                if 0 <= nr < R and 0 <= nc < C and (grid[nr][nc] == grid[r][c]) and lookup[ni]:
                    uf.union_set(i, ni)

        def check(i):
            if False:
                return 10
            (r, c) = divmod(i, C)
            if grid[r][c] == 0:
                return False
            lookup = set()
            for (dr, dc) in DIRECTIONS:
                (nr, nc) = (r + dr, c + dc)
                if 0 <= nr < R and 0 <= nc < C and (grid[nr][nc] == grid[r][c]):
                    lookup.add(uf.find_set(nr * C + nc))
            return len(lookup) != 1

        def dfs(left, right):
            if False:
                print('Hello World!')
            if left == right:
                return check(left)
            mid = left + (right - left) // 2
            (l1, r1, l2, r2) = (left, mid, mid + 1, right)
            for _ in xrange(2):
                uf.snapshot()
                for i in xrange(l1, r1 + 1):
                    lookup[i] = True
                    merge(i)
                if dfs(l2, r2):
                    return True
                for i in xrange(l1, r1 + 1):
                    lookup[i] = False
                uf.rollback()
                (l1, r1, l2, r2) = (l2, r2, l1, r1)
            return False
        (R, C) = (len(grid), len(grid[0]))
        if count_islands(grid) != 1:
            return 0
        uf = PersistentUnionFind(R * C)
        lookup = [False] * (R * C)
        return 1 if dfs(0, R * C - 1) else 2

class Solution3(object):

    def minDays(self, grid):
        if False:
            while True:
                i = 10
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '
        DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def floodfill(grid, i, j, lookup):
            if False:
                i = 10
                return i + 15
            stk = [(i, j)]
            lookup[i][j] = 1
            while stk:
                (i, j) = stk.pop()
                for (di, dj) in DIRECTIONS:
                    (ni, nj) = (i + di, j + dj)
                    if not (0 <= ni < R and 0 <= nj < C and grid[ni][nj] and (not lookup[ni][nj])):
                        continue
                    lookup[ni][nj] = 1
                    stk.append((ni, nj))

        def count_islands(grid):
            if False:
                return 10
            lookup = [[0] * C for _ in xrange(R)]
            island_cnt = 0
            for i in xrange(R):
                for j in xrange(C):
                    if grid[i][j] == 0 or lookup[i][j]:
                        continue
                    island_cnt += 1
                    floodfill(grid, i, j, lookup)
            return island_cnt
        (R, C) = (len(grid), len(grid[0]))
        if count_islands(grid) != 1:
            return 0
        for i in xrange(R):
            for j in xrange(C):
                if grid[i][j] == 0:
                    continue
                grid[i][j] = 0
                island_cnt = count_islands(grid)
                grid[i][j] = 1
                if island_cnt != 1:
                    return 1
        return 2