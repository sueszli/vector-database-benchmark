import bisect

class Solution(object):

    def longestObstacleCourseAtEachPosition(self, obstacles):
        if False:
            while True:
                i = 10
        '\n        :type obstacles: List[int]\n        :rtype: List[int]\n        '
        (result, stk) = ([], [])
        for x in obstacles:
            i = bisect.bisect_right(stk, x)
            result.append(i + 1)
            if i == len(stk):
                stk.append(0)
            stk[i] = x
        return result

class SegmentTree(object):

    def __init__(self, N, build_fn=lambda x, y: [y] * (2 * x), query_fn=lambda x, y: y if x is None else max(x, y), update_fn=lambda x, y: y, default_val=0):
        if False:
            i = 10
            return i + 15
        self.N = N
        self.H = (N - 1).bit_length()
        self.query_fn = query_fn
        self.update_fn = update_fn
        self.default_val = default_val
        self.tree = build_fn(N, default_val)
        self.lazy = [None] * N

    def __apply(self, x, val):
        if False:
            for i in range(10):
                print('nop')
        self.tree[x] = self.update_fn(self.tree[x], val)
        if x < self.N:
            self.lazy[x] = self.update_fn(self.lazy[x], val)

    def update(self, L, R, h):
        if False:
            print('Hello World!')

        def pull(x):
            if False:
                for i in range(10):
                    print('nop')
            while x > 1:
                x //= 2
                self.tree[x] = self.query_fn(self.tree[x * 2], self.tree[x * 2 + 1])
                if self.lazy[x] is not None:
                    self.tree[x] = self.update_fn(self.tree[x], self.lazy[x])
        L += self.N
        R += self.N
        (L0, R0) = (L, R)
        while L <= R:
            if L & 1:
                self.__apply(L, h)
                L += 1
            if R & 1 == 0:
                self.__apply(R, h)
                R -= 1
            L //= 2
            R //= 2
        pull(L0)
        pull(R0)

    def query(self, L, R):
        if False:
            return 10

        def push(x):
            if False:
                for i in range(10):
                    print('nop')
            n = 2 ** self.H
            while n != 1:
                y = x // n
                if self.lazy[y] is not None:
                    self.__apply(y * 2, self.lazy[y])
                    self.__apply(y * 2 + 1, self.lazy[y])
                    self.lazy[y] = None
                n //= 2
        result = None
        if L > R:
            return result
        L += self.N
        R += self.N
        push(L)
        push(R)
        while L <= R:
            if L & 1:
                result = self.query_fn(result, self.tree[L])
                L += 1
            if R & 1 == 0:
                result = self.query_fn(result, self.tree[R])
                R -= 1
            L //= 2
            R //= 2
        return result

    def __str__(self):
        if False:
            i = 10
            return i + 15
        showList = []
        for i in xrange(self.N):
            showList.append(self.query(i, i))
        return ','.join(map(str, showList))

class Solution2_TLE(object):

    def longestObstacleCourseAtEachPosition(self, obstacles):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type obstacles: List[int]\n        :rtype: List[int]\n        '
        sorted_obstacles = sorted(set(obstacles))
        lookup = {x: i for (i, x) in enumerate(sorted_obstacles)}
        segment_tree = SegmentTree(len(lookup))
        result = []
        for x in obstacles:
            cnt = segment_tree.query(0, lookup[x]) + 1
            result.append(cnt)
            segment_tree.update(lookup[x], lookup[x], cnt)
        return result