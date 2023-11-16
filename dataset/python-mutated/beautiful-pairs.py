import random
import itertools
import math
random.seed(0)

class Solution(object):

    def beautifulPair(self, nums1, nums2):
        if False:
            print('Hello World!')
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: List[int]\n        '
        INF = float('inf')

        def dist(a, b):
            if False:
                return 10
            if a[2] > b[2]:
                (a, b) = (b, a)
            return [abs(a[0] - b[0]) + abs(a[1] - b[1]), a[2], b[2]]

        def cell(point, size):
            if False:
                while True:
                    i = 10
            (x, y, _) = point
            return (math.floor(x / size), math.floor(y / size))

        def improve():
            if False:
                return 10
            lookup = {}
            for p in points:
                (i, j) = map(int, cell(p, result[0] / 2.0))
                for ni in xrange(i - 2, i + 2 + 1):
                    for nj in xrange(j - 2, j + 2 + 1):
                        if (ni, nj) not in lookup:
                            continue
                        d = dist(p, lookup[ni, nj])
                        if d < result:
                            result[:] = d
                            return True
                lookup[i, j] = p
            return False
        points = [(i, j, idx) for (idx, (i, j)) in enumerate(itertools.izip(nums1, nums2))]
        result = [INF] * 3
        lookup = {}
        for i in reversed(xrange(len(points))):
            if points[i][:2] in lookup:
                result = [0, i, lookup[points[i][:2]]]
            lookup[points[i][:2]] = i
        if result[0] == 0:
            return result[1:]
        random.shuffle(points)
        result = dist(points[0], points[1])
        while improve():
            pass
        return result[1:]
import itertools

class Solution2(object):

    def beautifulPair(self, nums1, nums2):
        if False:
            return 10
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: List[int]\n        '
        INF = float('inf')
        MAX_NEIGHBOR_COUNT = (8 + 2) // 2

        def dist(a, b):
            if False:
                while True:
                    i = 10
            if a > b:
                (a, b) = (b, a)
            return [abs(points[a][0] - points[b][0]) + abs(points[a][1] - points[b][1]), a, b]

        def merge_sort(left, right):
            if False:
                i = 10
                return i + 15

            def update(arr, i):
                if False:
                    for i in range(10):
                        print('nop')
                for j in reversed(xrange(len(arr))):
                    if points[i][1] - points[arr[j]][1] > result[0]:
                        break
                    result[:] = min(result, dist(i, arr[j]))
                else:
                    j = -1
                assert len(arr) - 1 - j <= MAX_NEIGHBOR_COUNT
            if left == right:
                return
            mid = left + (right - left) // 2
            x = points[order[mid]][0]
            merge_sort(left, mid)
            merge_sort(mid + 1, right)
            (tmp, tmp_l, tmp_r) = ([], [], [])
            (l, r) = (left, mid + 1)
            while l <= mid or r <= right:
                if r == right + 1 or (l <= mid and points[order[l]][1] <= points[order[r]][1]):
                    update(tmp_r, order[l])
                    if x - points[order[l]][0] <= result[0]:
                        tmp_l.append(order[l])
                    tmp.append(order[l])
                    l += 1
                else:
                    update(tmp_l, order[r])
                    if points[order[r]][0] - x <= result[0]:
                        tmp_r.append(order[r])
                    tmp.append(order[r])
                    r += 1
            order[left:right + 1] = tmp
        points = [(i, j) for (i, j) in itertools.izip(nums1, nums2)]
        result = [INF] * 3
        lookup = {}
        for i in reversed(xrange(len(points))):
            if points[i] in lookup:
                result = [0, (i, lookup[points[i]])]
            lookup[points[i]] = i
        if result[0] == 0:
            return result[1]
        order = range(len(points))
        order.sort(key=lambda x: points[x][0])
        merge_sort(0, len(points) - 1)
        return result[1:]
import itertools

class Solution3(object):

    def beautifulPair(self, nums1, nums2):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: List[int]\n        '
        INF = float('inf')
        MAX_NEIGHBOR_COUNT = 8

        def dist(a, b):
            if False:
                while True:
                    i = 10
            if a > b:
                (a, b) = (b, a)
            return [abs(points[a][0] - points[b][0]) + abs(points[a][1] - points[b][1]), a, b]

        def merge_sort(left, right):
            if False:
                while True:
                    i = 10
            if left == right:
                return
            mid = left + (right - left) // 2
            x = points[order[mid]][0]
            merge_sort(left, mid)
            merge_sort(mid + 1, right)
            r = mid + 1
            tmp = []
            for l in xrange(left, mid + 1):
                while r <= right and points[order[r]][1] < points[order[l]][1]:
                    tmp.append(order[r])
                    r += 1
                tmp.append(order[l])
            order[left:left + len(tmp)] = tmp
            stripe = [order[i] for i in xrange(left, right + 1) if abs(points[order[i]][0] - x) <= result[0]]
            for i in xrange(len(stripe) - 1):
                for j in xrange(i + 1, len(stripe)):
                    (x, y) = (stripe[i], stripe[j])
                    if points[y][1] - points[x][1] > result[0]:
                        break
                    result[:] = min(result, dist(x, y))
                else:
                    j = len(stripe)
                assert j - (i + 1) <= MAX_NEIGHBOR_COUNT
        points = [(i, j) for (i, j) in itertools.izip(nums1, nums2)]
        result = [INF] * 3
        lookup = {}
        for i in reversed(xrange(len(points))):
            if points[i] in lookup:
                result = [0, (i, lookup[points[i]])]
            lookup[points[i]] = i
        if result[0] == 0:
            return result[1]
        order = range(len(points))
        order.sort(key=lambda x: points[x][0])
        merge_sort(0, len(points) - 1)
        return result[1:]
import itertools

class Solution4(object):

    def beautifulPair(self, nums1, nums2):
        if False:
            print('Hello World!')
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: List[int]\n        '
        INF = float('inf')

        class SegmentTree(object):

            def __init__(self, N, build_fn=lambda _: [-INF, -INF], query_fn=lambda x, y: y if x is None else x if y is None else max(x, y), update_fn=lambda x: x):
                if False:
                    i = 10
                    return i + 15
                self.tree = [None] * (2 * 2 ** (N - 1).bit_length())
                self.base = len(self.tree) // 2
                self.query_fn = query_fn
                self.update_fn = update_fn
                for i in xrange(self.base, self.base + N):
                    self.tree[i] = build_fn(i - self.base)
                for i in reversed(xrange(1, self.base)):
                    self.tree[i] = query_fn(self.tree[2 * i], self.tree[2 * i + 1])

            def update(self, i, h):
                if False:
                    print('Hello World!')
                x = self.base + i
                self.tree[x] = self.update_fn(h)
                while x > 1:
                    x //= 2
                    self.tree[x] = self.query_fn(self.tree[x * 2], self.tree[x * 2 + 1])

            def query(self, L, R):
                if False:
                    i = 10
                    return i + 15
                if L > R:
                    return [-INF, -INF]
                L += self.base
                R += self.base
                left = right = None
                while L <= R:
                    if L & 1:
                        left = self.query_fn(left, self.tree[L])
                        L += 1
                    if R & 1 == 0:
                        right = self.query_fn(self.tree[R], right)
                        R -= 1
                    L //= 2
                    R //= 2
                return self.query_fn(left, right)

        def dist(a, b):
            if False:
                print('Hello World!')
            if a > b:
                (a, b) = (b, a)
            return [abs(points[a][0] - points[b][0]) + abs(points[a][1] - points[b][1]), a, b]
        points = [(i, j) for (i, j) in itertools.izip(nums1, nums2)]
        result = [INF] * 3
        lookup = {}
        for i in reversed(xrange(len(points))):
            if points[i] in lookup:
                result = [0, (i, lookup[points[i]])]
            lookup[points[i]] = i
        if result[0] == 0:
            return result[1]
        order = range(len(points))
        order.sort(key=lambda x: points[x][0])
        y_set = set((y for (_, y) in points))
        y_to_idx = {y: i for (i, y) in enumerate(sorted(y_set))}
        (st1, st2) = (SegmentTree(len(y_to_idx)), SegmentTree(len(y_to_idx)))
        for i in order:
            j = -st1.query(0, y_to_idx[points[i][1]] - 1)[1]
            if j != INF:
                assert points[j][1] < points[i][1]
                result = min(result, dist(i, j))
            st1.update(y_to_idx[points[i][1]], [points[i][0] + points[i][1], -i])
            j = -st2.query(y_to_idx[points[i][1]], len(y_to_idx) - 1)[1]
            if j != INF:
                assert points[j][1] >= points[i][1]
                result = min(result, dist(i, j))
            st2.update(y_to_idx[points[i][1]], [points[i][0] - points[i][1], -i])
        return result[1:]