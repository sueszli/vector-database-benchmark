import itertools
from sortedcontainers import SortedList

class Solution(object):

    def maxProfit(self, prices, profits):
        if False:
            return 10
        '\n        :type prices: List[int]\n        :type profits: List[int]\n        :rtype: int\n        '
        NEG_INF = float('-inf')

        def query(sl, k):
            if False:
                while True:
                    i = 10
            j = sl.bisect_left((k,))
            return sl[j - 1][1] if j - 1 >= 0 else NEG_INF

        def update(sl, k, v):
            if False:
                while True:
                    i = 10
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
        result = NEG_INF
        (sl1, sl2) = (SortedList(), SortedList())
        for (price, profit) in itertools.izip(prices, profits):
            result = max(result, query(sl2, price) + profit)
            update(sl1, price, profit)
            update(sl2, price, query(sl1, price) + profit)
        return result if result != NEG_INF else -1
from sortedcontainers import SortedList

class Solution2(object):

    def maxProfit(self, prices, profits):
        if False:
            print('Hello World!')
        '\n        :type prices: List[int]\n        :type profits: List[int]\n        :rtype: int\n        '
        NEG_INF = float('-inf')
        right = [NEG_INF] * len(prices)
        sl = SortedList()
        for i in reversed(xrange(len(prices))):
            j = sl.bisect_left((-prices[i],))
            if j - 1 >= 0:
                right[i] = sl[j - 1][1]
            if not (j - 1 < 0 or sl[j - 1][1] < profits[i]):
                continue
            sl.add((-prices[i], profits[i]))
            j = sl.bisect_left((-prices[i], profits[i]))
            while j + 1 < len(sl) and sl[j + 1][1] <= sl[j][1]:
                del sl[j + 1]
        result = NEG_INF
        sl = SortedList()
        for i in xrange(len(prices)):
            j = sl.bisect_left((prices[i],))
            if j - 1 >= 0:
                result = max(result, sl[j - 1][1] + profits[i] + right[i])
            if not (j - 1 < 0 or sl[j - 1][1] < profits[i]):
                continue
            sl.add((prices[i], profits[i]))
            j = sl.bisect_left((prices[i], profits[i]))
            while j + 1 < len(sl) and sl[j + 1][1] <= sl[j][1]:
                del sl[j + 1]
        return result if result != NEG_INF else -1
import itertools

class Solution3(object):

    def maxProfit(self, prices, profits):
        if False:
            i = 10
            return i + 15
        '\n        :type prices: List[int]\n        :type profits: List[int]\n        :rtype: int\n        '
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
                    print('Hello World!')
                i += 1
                while i < len(self.__bit):
                    self.__bit[i] = self.__fn(self.__bit[i], val)
                    i += i & -i

            def query(self, i):
                if False:
                    while True:
                        i = 10
                i += 1
                ret = self.__default
                while i > 0:
                    ret = self.__fn(ret, self.__bit[i])
                    i -= i & -i
                return ret
        price_to_idx = {x: i for (i, x) in enumerate(sorted(set(prices)))}
        result = NEG_INF
        (bit1, bit2) = (BIT(len(price_to_idx), default=NEG_INF, fn=max), BIT(len(price_to_idx), default=NEG_INF, fn=max))
        for (price, profit) in itertools.izip(prices, profits):
            result = max(result, bit2.query(price_to_idx[price] - 1) + profit)
            bit1.update(price_to_idx[price], profit)
            bit2.update(price_to_idx[price], bit1.query(price_to_idx[price] - 1) + profit)
        return result if result != NEG_INF else -1
import itertools

class Solution4(object):

    def maxProfit(self, prices, profits):
        if False:
            return 10
        '\n        :type prices: List[int]\n        :type profits: List[int]\n        :rtype: int\n        '
        NEG_INF = float('-inf')

        class SegmentTree(object):

            def __init__(self, N, build_fn=lambda _: None, query_fn=lambda x, y: max(x, y), update_fn=lambda x, y: max(x, y)):
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
                    while True:
                        i = 10
                x = self.base + i
                self.tree[x] = self.update_fn(self.tree[x], h)
                while x > 1:
                    x //= 2
                    self.tree[x] = self.query_fn(self.tree[x * 2], self.tree[x * 2 + 1])

            def query(self, L, R):
                if False:
                    return 10
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
        price_to_idx = {x: i for (i, x) in enumerate(sorted(set(prices)))}
        result = NEG_INF
        (st1, st2) = (SegmentTree(len(price_to_idx)), SegmentTree(len(price_to_idx)))
        for (price, profit) in itertools.izip(prices, profits):
            mx2 = st2.query(0, price_to_idx[price] - 1)
            if mx2 is not None:
                result = max(result, mx2 + profit)
            st1.update(price_to_idx[price], profit)
            mx1 = st1.query(0, price_to_idx[price] - 1)
            if mx1 is not None:
                st2.update(price_to_idx[price], mx1 + profit)
        return result if result != NEG_INF else -1

class Solution5(object):

    def maxProfit(self, prices, profits):
        if False:
            return 10
        '\n        :type prices: List[int]\n        :type profits: List[int]\n        :rtype: int\n        '
        NEG_INF = float('-inf')

        class SegmentTree(object):

            def __init__(self, N, build_fn=lambda _: None, query_fn=lambda x, y: max(x, y), update_fn=lambda x, y: max(x, y)):
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
                    return 10
                x = self.base + i
                self.tree[x] = self.update_fn(self.tree[x], h)
                while x > 1:
                    x //= 2
                    self.tree[x] = self.query_fn(self.tree[x * 2], self.tree[x * 2 + 1])

            def query(self, L, R):
                if False:
                    while True:
                        i = 10
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
        price_to_idx = {x: i for (i, x) in enumerate(sorted(set(prices)))}
        right = [NEG_INF] * len(prices)
        st = SegmentTree(len(price_to_idx))
        for i in reversed(xrange(len(prices))):
            right[i] = st.query(price_to_idx[prices[i]] + 1, len(price_to_idx) - 1)
            st.update(price_to_idx[prices[i]], profits[i])
        result = NEG_INF
        st = SegmentTree(len(price_to_idx))
        for i in xrange(len(prices)):
            left = st.query(0, price_to_idx[prices[i]] - 1)
            if left is not None and right[i] is not None:
                result = max(result, left + profits[i] + right[i])
            st.update(price_to_idx[prices[i]], profits[i])
        return result if result != NEG_INF else -1