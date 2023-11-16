class SegmentTree(object):

    def __init__(self, N, build_fn=lambda x, y: [y] * (2 * x), query_fn=lambda x, y: y if x is None else max(x, y), update_fn=lambda x, y: y if x is None else x + y, default_val=0):
        if False:
            return 10
        self.N = N
        self.H = (N - 1).bit_length()
        self.query_fn = query_fn
        self.update_fn = update_fn
        self.default_val = default_val
        self.tree = build_fn(N, default_val)
        self.lazy = [None] * N

    def __apply(self, x, val):
        if False:
            while True:
                i = 10
        self.tree[x] = self.update_fn(self.tree[x], val)
        if x < self.N:
            self.lazy[x] = self.update_fn(self.lazy[x], val)

    def update(self, L, R, h):
        if False:
            for i in range(10):
                print('nop')

        def pull(x):
            if False:
                while True:
                    i = 10
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
                i = 10
                return i + 15
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

class Solution(object):

    def minDayskVariants(self, points, k):
        if False:
            return 10
        '\n        :type points: List[List[int]]\n        :type k: int\n        :rtype: int\n        '

        def add_rec(rec, intervals):
            if False:
                for i in range(10):
                    print('nop')
            (x0, y0, x1, y1) = rec
            intervals.append([[x0, +1], [y0, y1]])
            intervals.append([[x1 + 1, -1], [y0, y1]])

        def check(points, k, l):
            if False:
                i = 10
                return i + 15
            intervals = []
            y_set = set()
            for (x, y) in points:
                add_rec([x - l, y - l, x + l, y + l], intervals)
                y_set.add(y - l)
                y_set.add(y + l)
            intervals.sort()
            y_to_idx = {y: i for (i, y) in enumerate(sorted(y_set))}
            st = SegmentTree(len(y_to_idx))
            for ([_, v], [y0, y1]) in intervals:
                st.update(y_to_idx[y0], y_to_idx[y1], v)
                if st.query(0, len(y_to_idx) - 1) >= k:
                    return True
            return False
        points = [[x + y, x - y] for (x, y) in points]
        min_x = min(points)[0]
        max_x = max(points)[0]
        min_y = min(points, key=lambda x: x[1])[1]
        max_y = max(points, key=lambda x: x[1])[1]
        (left, right) = (0, (max_x - min_x + (max_y - min_y) + 1) // 2)
        while left <= right:
            mid = left + (right - left) // 2
            if check(points, k, mid):
                right = mid - 1
            else:
                left = mid + 1
        return left
import collections

class Solution2(object):

    def minDayskVariants(self, points, k):
        if False:
            while True:
                i = 10
        '\n        :type points: List[List[int]]\n        :type k: int\n        :rtype: int\n        '

        def add_rec(rec, intervals):
            if False:
                while True:
                    i = 10
            (x0, y0, x1, y1) = rec
            intervals[x0][y0] += 1
            intervals[x0][y1 + 1] -= 1
            intervals[x1 + 1][y0] -= 1
            intervals[x1 + 1][y1 + 1] += 1

        def check(points, k, l):
            if False:
                i = 10
                return i + 15
            intervals = collections.defaultdict(lambda : collections.defaultdict(int))
            y_set = set()
            for (x, y) in points:
                add_rec([x - l, y - l, x + l, y + l], intervals)
                y_set.add(y - l)
                y_set.add(y + l + 1)
            sorted_y = sorted(y_set)
            sorted_x = sorted(intervals.iterkeys())
            count = collections.Counter()
            for x in sorted_x:
                for (y, c) in intervals[x].iteritems():
                    count[y] += c
                cnt = 0
                for y in sorted_y:
                    cnt += count[y]
                    if cnt >= k:
                        return True
            return False
        points = [[x + y, x - y] for (x, y) in points]
        min_x = min(points)[0]
        max_x = max(points)[0]
        min_y = min(points, key=lambda x: x[1])[1]
        max_y = max(points, key=lambda x: x[1])[1]
        (left, right) = (0, (max_x - min_x + (max_y - min_y) + 1) // 2)
        while left <= right:
            mid = left + (right - left) // 2
            if check(points, k, mid):
                right = mid - 1
            else:
                left = mid + 1
        return left