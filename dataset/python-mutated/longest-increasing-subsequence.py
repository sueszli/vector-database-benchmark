import bisect

class Solution(object):

    def lengthOfLIS(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        LIS = []

        def insert(target):
            if False:
                print('Hello World!')
            left = bisect.bisect_left(LIS, target)
            if left == len(LIS):
                LIS.append(target)
            else:
                LIS[left] = target
        for num in nums:
            insert(num)
        return len(LIS)

class Solution2(object):

    def lengthOfLIS(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        LIS = []

        def insert(target):
            if False:
                print('Hello World!')
            (left, right) = (0, len(LIS) - 1)
            while left <= right:
                mid = left + (right - left) // 2
                if LIS[mid] >= target:
                    right = mid - 1
                else:
                    left = mid + 1
            if left == len(LIS):
                LIS.append(target)
            else:
                LIS[left] = target
        for num in nums:
            insert(num)
        return len(LIS)

class SegmentTree(object):

    def __init__(self, N, build_fn=lambda x, y: [y] * (2 * x), query_fn=lambda x, y: y if x is None else max(x, y), update_fn=lambda x, y: y, default_val=0):
        if False:
            while True:
                i = 10
        self.N = N
        self.H = (N - 1).bit_length()
        self.query_fn = query_fn
        self.update_fn = update_fn
        self.default_val = default_val
        self.tree = build_fn(N, default_val)
        self.lazy = [None] * N

    def __apply(self, x, val):
        if False:
            return 10
        self.tree[x] = self.update_fn(self.tree[x], val)
        if x < self.N:
            self.lazy[x] = self.update_fn(self.lazy[x], val)

    def update(self, L, R, h):
        if False:
            print('Hello World!')

        def pull(x):
            if False:
                return 10
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
            print('Hello World!')

        def push(x):
            if False:
                while True:
                    i = 10
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
            print('Hello World!')
        showList = []
        for i in xrange(self.N):
            showList.append(self.query(i, i))
        return ','.join(map(str, showList))

class Solution3(object):

    def lengthOfLIS(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        sorted_nums = sorted(set(nums))
        lookup = {num: i for (i, num) in enumerate(sorted_nums)}
        segment_tree = SegmentTree(len(lookup))
        for num in nums:
            segment_tree.update(lookup[num], lookup[num], segment_tree.query(0, lookup[num] - 1) + 1 if lookup[num] >= 1 else 1)
        return segment_tree.query(0, len(lookup) - 1) if len(lookup) >= 1 else 0

class Solution4(object):

    def lengthOfLIS(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        dp = []
        for i in xrange(len(nums)):
            dp.append(1)
            for j in xrange(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp) if dp else 0