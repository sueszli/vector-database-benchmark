class Solution(object):

    def earliestAndLatest(self, n, firstPlayer, secondPlayer):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type firstPlayer: int\n        :type secondPlayer: int\n        :rtype: List[int]\n        '

        def memoization(t, l, r, lookup):
            if False:
                i = 10
                return i + 15
            if (t, l, r) not in lookup:
                if l == r:
                    return (1, 1)
                if l > r:
                    (l, r) = (r, l)
                result = [float('inf'), 0]
                for i in xrange(l + 1):
                    (l_win_cnt, l_lose_cnt, nt, pair_cnt) = (i + 1, l - i, (t + 1) // 2, t // 2)
                    min_j = max(l_lose_cnt, r - (pair_cnt - l_lose_cnt))
                    max_j = min(r - l_win_cnt, nt - l_win_cnt - 1)
                    for j in xrange(min_j, max_j + 1):
                        tmp = memoization(nt, i, j, lookup)
                        result = (min(result[0], tmp[0] + 1), max(result[1], tmp[1] + 1))
                lookup[t, l, r] = result
            return lookup[t, l, r]
        return memoization(n, firstPlayer - 1, n - secondPlayer, {})