import heapq

class Solution(object):

    def minimumEffortPath(self, heights):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type heights: List[List[int]]\n        :rtype: int\n        '
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dst = (len(heights) - 1, len(heights[0]) - 1)
        dist = [[float('inf')] * len(heights[0]) for _ in xrange(len(heights))]
        dist[0][0] = 0
        min_heap = [(0, 0, 0)]
        lookup = [[False] * len(heights[0]) for _ in xrange(len(heights))]
        while min_heap:
            (d, r, c) = heapq.heappop(min_heap)
            if lookup[r][c]:
                continue
            lookup[r][c] = True
            if (r, c) == dst:
                return d
            for (dr, dc) in directions:
                (nr, nc) = (r + dr, c + dc)
                if not (0 <= nr < len(heights) and 0 <= nc < len(heights[0]) and (not lookup[nr][nc])):
                    continue
                nd = max(d, abs(heights[nr][nc] - heights[r][c]))
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    heapq.heappush(min_heap, (nd, nr, nc))
        return -1
import collections

class UnionFind(object):

    def __init__(self, n):
        if False:
            print('Hello World!')
        self.set = range(n)
        self.rank = [0] * n

    def find_set(self, x):
        if False:
            print('Hello World!')
        stk = []
        while self.set[x] != x:
            stk.append(x)
            x = self.set[x]
        while stk:
            self.set[stk.pop()] = x
        return x

    def union_set(self, x, y):
        if False:
            while True:
                i = 10
        (x_root, y_root) = map(self.find_set, (x, y))
        if x_root == y_root:
            return False
        if self.rank[x_root] < self.rank[y_root]:
            self.set[x_root] = y_root
        elif self.rank[x_root] > self.rank[y_root]:
            self.set[y_root] = x_root
        else:
            self.set[y_root] = x_root
            self.rank[x_root] += 1
        return True

class Solution2(object):

    def minimumEffortPath(self, heights):
        if False:
            while True:
                i = 10
        '\n        :type heights: List[List[int]]\n        :rtype: int\n        '

        def index(n, i, j):
            if False:
                while True:
                    i = 10
            return i * n + j
        diffs = []
        for i in xrange(len(heights)):
            for j in xrange(len(heights[0])):
                if i > 0:
                    diffs.append((abs(heights[i][j] - heights[i - 1][j]), index(len(heights[0]), i - 1, j), index(len(heights[0]), i, j)))
                if j > 0:
                    diffs.append((abs(heights[i][j] - heights[i][j - 1]), index(len(heights[0]), i, j - 1), index(len(heights[0]), i, j)))
        diffs.sort()
        union_find = UnionFind(len(heights) * len(heights[0]))
        for (d, i, j) in diffs:
            if union_find.union_set(i, j):
                if union_find.find_set(index(len(heights[0]), 0, 0)) == union_find.find_set(index(len(heights[0]), len(heights) - 1, len(heights[0]) - 1)):
                    return d
        return 0

class Solution3(object):

    def minimumEffortPath(self, heights):
        if False:
            while True:
                i = 10
        '\n        :type heights: List[List[int]]\n        :rtype: int\n        '
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def check(heights, x):
            if False:
                while True:
                    i = 10
            lookup = [[False] * len(heights[0]) for _ in xrange(len(heights))]
            (left, right) = ({(0, 0)}, {(len(heights) - 1, len(heights[0]) - 1)})
            while left:
                for (r, c) in left:
                    lookup[r][c] = True
                new_left = set()
                for (r, c) in left:
                    if (r, c) in right:
                        return True
                    for (dr, dc) in directions:
                        (nr, nc) = (r + dr, c + dc)
                        if not (0 <= nr < len(heights) and 0 <= nc < len(heights[0]) and (abs(heights[nr][nc] - heights[r][c]) <= x) and (not lookup[nr][nc])):
                            continue
                        new_left.add((nr, nc))
                left = new_left
                if len(left) > len(right):
                    (left, right) = (right, left)
            return False
        (left, right) = (0, 10 ** 6)
        while left <= right:
            mid = left + (right - left) // 2
            if check(heights, mid):
                right = mid - 1
            else:
                left = mid + 1
        return left
import collections

class Solution4(object):

    def minimumEffortPath(self, heights):
        if False:
            print('Hello World!')
        '\n        :type heights: List[List[int]]\n        :rtype: int\n        '
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def check(heights, x):
            if False:
                for i in range(10):
                    print('nop')
            lookup = [[False] * len(heights[0]) for _ in xrange(len(heights))]
            q = collections.deque([(0, 0)])
            while q:
                (r, c) = q.popleft()
                if (r, c) == (len(heights) - 1, len(heights[0]) - 1):
                    return True
                for (dr, dc) in directions:
                    (nr, nc) = (r + dr, c + dc)
                    if not (0 <= nr < len(heights) and 0 <= nc < len(heights[0]) and (abs(heights[nr][nc] - heights[r][c]) <= x) and (not lookup[nr][nc])):
                        continue
                    lookup[nr][nc] = True
                    q.append((nr, nc))
            return False
        (left, right) = (0, 10 ** 6)
        while left <= right:
            mid = left + (right - left) // 2
            if check(heights, mid):
                right = mid - 1
            else:
                left = mid + 1
        return left

class Solution5(object):

    def minimumEffortPath(self, heights):
        if False:
            while True:
                i = 10
        '\n        :type heights: List[List[int]]\n        :rtype: int\n        '
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def check(heights, x):
            if False:
                for i in range(10):
                    print('nop')
            lookup = [[False] * len(heights[0]) for _ in xrange(len(heights))]
            stk = [(0, 0)]
            while stk:
                (r, c) = stk.pop()
                if (r, c) == (len(heights) - 1, len(heights[0]) - 1):
                    return True
                for (dr, dc) in directions:
                    (nr, nc) = (r + dr, c + dc)
                    if not (0 <= nr < len(heights) and 0 <= nc < len(heights[0]) and (abs(heights[nr][nc] - heights[r][c]) <= x) and (not lookup[nr][nc])):
                        continue
                    lookup[nr][nc] = True
                    stk.append((nr, nc))
            return False
        (left, right) = (0, 10 ** 6)
        while left <= right:
            mid = left + (right - left) // 2
            if check(heights, mid):
                right = mid - 1
            else:
                left = mid + 1
        return left