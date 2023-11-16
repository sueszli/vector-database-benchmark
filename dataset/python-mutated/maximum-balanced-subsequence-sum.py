from sortedcontainers import SortedList

class Solution(object):

    def maxBalancedSubsequenceSum(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        NEG_INF = float('-inf')

        def query(sl, k):
            if False:
                print('Hello World!')
            j = sl.bisect_left((k,))
            return sl[j - 1][1] if j - 1 >= 0 else NEG_INF

        def update(sl, k, v):
            if False:
                return 10
            j = sl.bisect_left((k,))
            if j < len(sl) and sl[j][0] == k:
                if not sl[j][1] < v:
                    return
                del sl[j]
            elif not (j - 1 < 0 or sl[j - 1][1] < v):
                return
            sl.add((k, v))
            while j + 1 < len(sl) and sl[j + 1][1] <= sl[j][1]:
                del sl[j + 1]
        sl = SortedList()
        for (i, x) in enumerate(nums):
            v = max(query(sl, x - i + 1), 0) + x
            update(sl, x - i, v)
        return sl[-1][1]

class Solution2(object):

    def maxBalancedSubsequenceSum(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        NEG_INF = float('-inf')

        class BIT(object):

            def __init__(self, n, default=0, fn=lambda x, y: x + y):
                if False:
                    return 10
                self.__bit = [NEG_INF] * (n + 1)
                self.__default = default
                self.__fn = fn

            def update(self, i, val):
                if False:
                    return 10
                i += 1
                while i < len(self.__bit):
                    self.__bit[i] = self.__fn(self.__bit[i], val)
                    i += i & -i

            def query(self, i):
                if False:
                    print('Hello World!')
                i += 1
                ret = self.__default
                while i > 0:
                    ret = self.__fn(ret, self.__bit[i])
                    i -= i & -i
                return ret
        val_to_idx = {x: i for (i, x) in enumerate(sorted({x - i for (i, x) in enumerate(nums)}))}
        bit = BIT(len(val_to_idx), default=NEG_INF, fn=max)
        for (i, x) in enumerate(nums):
            v = max(bit.query(val_to_idx[x - i]), 0) + x
            bit.update(val_to_idx[x - i], v)
        return bit.query(len(val_to_idx) - 1)

class Solution3(object):

    def maxBalancedSubsequenceSum(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        NEG_INF = float('-inf')

        class SegmentTree(object):

            def __init__(self, N, build_fn=lambda _: None, query_fn=lambda x, y: max(x, y), update_fn=lambda x, y: max(x, y)):
                if False:
                    while True:
                        i = 10
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
                    for i in range(10):
                        print('nop')
                x = self.base + i
                self.tree[x] = self.update_fn(self.tree[x], h)
                while x > 1:
                    x //= 2
                    self.tree[x] = self.query_fn(self.tree[x * 2], self.tree[x * 2 + 1])

            def query(self, L, R):
                if False:
                    i = 10
                    return i + 15
                if L > R:
                    return None
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
        val_to_idx = {x: i for (i, x) in enumerate(sorted({x - i for (i, x) in enumerate(nums)}))}
        st = SegmentTree(len(val_to_idx))
        for (i, x) in enumerate(nums):
            v = max(st.query(0, val_to_idx[x - i]), 0) + x
            st.update(val_to_idx[x - i], v)
        return st.query(0, len(val_to_idx) - 1)