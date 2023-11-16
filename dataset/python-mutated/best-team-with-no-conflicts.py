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
            print('Hello World!')
        self.tree[x] = self.update_fn(self.tree[x], val)
        if x < self.N:
            self.lazy[x] = self.update_fn(self.lazy[x], val)

    def update(self, L, R, h):
        if False:
            return 10

        def pull(x):
            if False:
                print('Hello World!')
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
                print('Hello World!')
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
            for i in range(10):
                print('nop')
        showList = []
        for i in xrange(self.N):
            showList.append(self.query(i, i))
        return ','.join(map(str, showList))

class Solution(object):

    def bestTeamScore(self, scores, ages):
        if False:
            return 10
        '\n        :type scores: List[int]\n        :type ages: List[int]\n        :rtype: int\n        '
        players = sorted(zip(scores, ages))
        sorted_ages = sorted(set(ages))
        lookup = {age: i for (i, age) in enumerate(sorted_ages)}
        segment_tree = SegmentTree(len(lookup))
        result = 0
        for (score, age) in players:
            segment_tree.update(lookup[age], lookup[age], segment_tree.query(0, lookup[age]) + score)
        return segment_tree.query(0, len(lookup) - 1)

class Solution2(object):

    def bestTeamScore(self, scores, ages):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type scores: List[int]\n        :type ages: List[int]\n        :rtype: int\n        '
        players = sorted(zip(ages, scores))
        sorted_scores = sorted(set(scores))
        lookup = {score: i for (i, score) in enumerate(sorted_scores)}
        segment_tree = SegmentTree(len(lookup))
        result = 0
        for (age, score) in players:
            segment_tree.update(lookup[score], lookup[score], segment_tree.query(0, lookup[score]) + score)
        return segment_tree.query(0, len(lookup) - 1)
import collections

class Solution3(object):

    def bestTeamScore(self, scores, ages):
        if False:
            while True:
                i = 10
        '\n        :type scores: List[int]\n        :type ages: List[int]\n        :rtype: int\n        '
        players = sorted(zip(scores, ages))
        sorted_ages = sorted(set(ages))
        dp = collections.defaultdict(int)
        result = 0
        for (score, age) in players:
            dp[age] = max((dp[a] for a in sorted_ages if a <= age)) + score
        return max(dp.itervalues())
import collections

class Solution4(object):

    def bestTeamScore(self, scores, ages):
        if False:
            while True:
                i = 10
        '\n        :type scores: List[int]\n        :type ages: List[int]\n        :rtype: int\n        '
        players = sorted(zip(ages, scores))
        sorted_scores = sorted(set(scores))
        dp = collections.defaultdict(int)
        result = 0
        for (age, score) in players:
            dp[score] = max((dp[s] for s in sorted_scores if s <= score)) + score
        return max(dp.itervalues())

class Solution5(object):

    def bestTeamScore(self, scores, ages):
        if False:
            print('Hello World!')
        '\n        :type scores: List[int]\n        :type ages: List[int]\n        :rtype: int\n        '
        players = sorted(zip(scores, ages))
        dp = [0] * len(players)
        result = 0
        for i in xrange(len(players)):
            dp[i] = players[i][0]
            for j in xrange(i):
                if players[j][1] <= players[i][1]:
                    dp[i] = max(dp[i], dp[j] + players[i][0])
            result = max(result, dp[i])
        return result

class Solution6(object):

    def bestTeamScore(self, scores, ages):
        if False:
            print('Hello World!')
        '\n        :type scores: List[int]\n        :type ages: List[int]\n        :rtype: int\n        '
        players = sorted(zip(ages, scores))
        dp = [0] * len(players)
        result = 0
        for i in xrange(len(players)):
            dp[i] = players[i][1]
            for j in xrange(i):
                if players[j][1] <= players[i][1]:
                    dp[i] = max(dp[i], dp[j] + players[i][1])
            result = max(result, dp[i])
        return result