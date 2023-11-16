import collections
import itertools

class Solution(object):

    def colorTheGrid(self, m, n):
        if False:
            print('Hello World!')
        '\n        :type m: int\n        :type n: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7

        def backtracking(mask1, mask2, basis, result):
            if False:
                while True:
                    i = 10
            if not basis:
                result.append(mask2)
                return
            for i in xrange(3):
                if (mask1 == -1 or mask1 // basis % 3 != i) and (mask2 == -1 or mask2 // (basis * 3) % 3 != i):
                    backtracking(mask1, mask2 + i * basis if mask2 != -1 else i * basis, basis // 3, result)

        def matrix_mult(A, B):
            if False:
                i = 10
                return i + 15
            ZB = zip(*B)
            return [[sum((a * b % MOD for (a, b) in itertools.izip(row, col))) % MOD for col in ZB] for row in A]

        def matrix_expo(A, K):
            if False:
                return 10
            result = [[int(i == j) for j in xrange(len(A))] for i in xrange(len(A))]
            while K:
                if K % 2:
                    result = matrix_mult(result, A)
                A = matrix_mult(A, A)
                K /= 2
            return result

        def normalize(basis, mask):
            if False:
                for i in range(10):
                    print('nop')
            norm = {}
            result = 0
            while basis:
                x = mask // basis % 3
                if x not in norm:
                    norm[x] = len(norm)
                result += norm[x] * basis
                basis //= 3
            return result
        if m > n:
            (m, n) = (n, m)
        basis = 3 ** (m - 1)
        masks = []
        backtracking(-1, -1, basis, masks)
        assert len(masks) == 3 * 2 ** (m - 1)
        lookup = {mask: normalize(basis, mask) for mask in masks}
        normalized_mask_cnt = collections.Counter((lookup[mask] for mask in masks))
        assert len(normalized_mask_cnt) == 3 * 2 ** (m - 1) // 3 // (2 if m >= 2 else 1)
        adj = collections.defaultdict(list)
        for mask in normalized_mask_cnt.iterkeys():
            backtracking(mask, -1, basis, adj[mask])
        normalized_adj = collections.defaultdict(lambda : collections.defaultdict(int))
        for (mask1, masks2) in adj.iteritems():
            for mask2 in masks2:
                normalized_adj[mask1][lookup[mask2]] = (normalized_adj[mask1][lookup[mask2]] + 1) % MOD
        assert 2 * 3 ** m // 3 // 2 // 3 <= sum((len(v) for v in normalized_adj.itervalues())) <= 2 * 3 ** m // 3 // 2
        return reduce(lambda x, y: (x + y) % MOD, matrix_mult([normalized_mask_cnt.values()], matrix_expo([[normalized_adj[mask1][mask2] for mask2 in normalized_mask_cnt.iterkeys()] for mask1 in normalized_mask_cnt.iterkeys()], n - 1))[0], 0)
import collections

class Solution2(object):

    def colorTheGrid(self, m, n):
        if False:
            print('Hello World!')
        '\n        :type m: int\n        :type n: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7

        def find_masks(m, basis):
            if False:
                i = 10
                return i + 15
            masks = [0]
            for c in xrange(m):
                new_masks = []
                for mask in masks:
                    choices = {0, 1, 2}
                    if c > 0:
                        choices.discard(mask // basis)
                    for x in choices:
                        new_masks.append(x * basis + mask // 3)
                masks = new_masks
            return masks

        def find_adj(m, basis, dp):
            if False:
                print('Hello World!')
            adj = collections.defaultdict(list)
            for mask in dp.iterkeys():
                adj[mask].append(mask)
            for c in xrange(m):
                assert sum((len(v) for v in adj.itervalues())) == (3 ** c * 2 ** (m - (c - 1)) if c >= 1 else 3 * 2 ** (m - 1)) // 3 // (2 if m >= 2 else 1)
                new_adj = collections.defaultdict(list)
                for (mask1, mask2s) in adj.iteritems():
                    for mask in mask2s:
                        choices = {0, 1, 2}
                        choices.discard(mask % 3)
                        if c > 0:
                            choices.discard(mask // basis)
                        for x in choices:
                            new_adj[mask1].append(x * basis + mask // 3)
                adj = new_adj
            assert sum((3 ** c * 2 ** (m - (c - 1)) if c >= 1 else 3 * 2 ** (m - 1) for c in xrange(m))) == 4 * 3 ** m - 9 * 2 ** (m - 1)
            return adj

        def normalize(basis, mask):
            if False:
                print('Hello World!')
            norm = {}
            result = 0
            while basis:
                x = mask // basis % 3
                if x not in norm:
                    norm[x] = len(norm)
                result += norm[x] * basis
                basis //= 3
            return result
        if m > n:
            (m, n) = (n, m)
        basis = 3 ** (m - 1)
        masks = find_masks(m, basis)
        assert len(masks) == 3 * 2 ** (m - 1)
        lookup = {mask: normalize(basis, mask) for mask in masks}
        dp = collections.Counter((lookup[mask] for mask in masks))
        adj = find_adj(m, basis, dp)
        normalized_adj = collections.defaultdict(lambda : collections.defaultdict(int))
        for (mask1, mask2s) in adj.iteritems():
            for mask2 in mask2s:
                normalized_adj[lookup[mask1]][lookup[mask2]] = (normalized_adj[lookup[mask1]][lookup[mask2]] + 1) % MOD
        assert 2 * 3 ** m // 3 // 2 // 3 <= sum((len(v) for v in normalized_adj.itervalues())) <= 2 * 3 ** m // 3 // 2
        for _ in xrange(n - 1):
            assert len(dp) == 3 * 2 ** (m - 1) // 3 // (2 if m >= 2 else 1)
            new_dp = collections.Counter()
            for (mask, v) in dp.iteritems():
                for (new_mask, cnt) in normalized_adj[mask].iteritems():
                    new_dp[lookup[new_mask]] = (new_dp[lookup[new_mask]] + v * cnt) % MOD
            dp = new_dp
        return reduce(lambda x, y: (x + y) % MOD, dp.itervalues(), 0)
import collections

class Solution3(object):

    def colorTheGrid(self, m, n):
        if False:
            while True:
                i = 10
        '\n        :type m: int\n        :type n: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7

        def normalize(basis, mask, lookup):
            if False:
                for i in range(10):
                    print('nop')
            if mask not in lookup[basis]:
                norm = {}
                (result, b) = (0, basis)
                while b:
                    x = mask // b % 3
                    if x not in norm:
                        norm[x] = len(norm)
                    result += norm[x] * b
                    b //= 3
                lookup[basis][mask] = result
            return lookup[basis][mask]
        if m > n:
            (m, n) = (n, m)
        basis = b = 3 ** (m - 1)
        lookup = collections.defaultdict(dict)
        dp = collections.Counter({0: 1})
        for idx in xrange(m * n):
            (r, c) = divmod(idx, m)
            assert r != 0 or c != 0 or len(dp) == 1
            assert r != 0 or c == 0 or len(dp) == 3 * 2 ** (c - 1) // 3 // (2 if c >= 2 else 1)
            assert r == 0 or c != 0 or len(dp) == 3 * 2 ** (m - 1) // 3 // (2 if m >= 2 else 1)
            assert r == 0 or c == 0 or len(dp) == (1 if m == 1 else 2 if m == 2 else 3 * 3 * 2 ** (m - 2) // 3 // 2)
            new_dp = collections.Counter()
            for (mask, v) in dp.iteritems():
                choices = {0, 1, 2}
                if r > 0:
                    choices.discard(mask % 3)
                if c > 0:
                    choices.discard(mask // basis)
                for x in choices:
                    new_mask = normalize(basis // b, (x * basis + mask // 3) // b, lookup) * b
                    new_dp[new_mask] = (new_dp[new_mask] + v) % MOD
            if b > 1:
                b //= 3
            dp = new_dp
        return reduce(lambda x, y: (x + y) % MOD, dp.itervalues(), 0)