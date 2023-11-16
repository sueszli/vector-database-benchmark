import bisect

class SegmentTree(object):

    def __init__(self, N, build_fn=lambda _: 0, query_fn=lambda x, y: y if x is None else x if y is None else max(x, y), update_fn=lambda x: x):
        if False:
            return 10
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
            return 10
        x = self.base + i
        self.tree[x] = self.update_fn(h)
        while x > 1:
            x //= 2
            self.tree[x] = self.query_fn(self.tree[x * 2], self.tree[x * 2 + 1])

    def query(self, L, R):
        if False:
            for i in range(10):
                print('nop')
        if L > R:
            return 0
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

class Solution(object):

    def lengthOfLIS(self, nums, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        sorted_nums = sorted({x - 1 for x in nums})
        num_to_idx = {x: i for (i, x) in enumerate(sorted_nums)}
        st = SegmentTree(len(num_to_idx))
        for x in nums:
            x -= 1
            st.update(num_to_idx[x], st.query(bisect.bisect_left(sorted_nums, x - k), num_to_idx[x] - 1) + 1)
        return st.tree[1]