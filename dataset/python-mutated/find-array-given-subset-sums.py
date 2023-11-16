class Solution(object):

    def recoverArray(self, n, sums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type sums: List[int]\n        :rtype: List[int]\n        '
        sums.sort()
        (shift, l) = (0, len(sums))
        result = []
        for _ in xrange(n):
            new_shift = sums[0] - sums[1]
            assert new_shift <= 0
            (has_zero, j, k) = (False, 0, 0)
            for i in xrange(l):
                if k < j and sums[k] == sums[i]:
                    k += 1
                else:
                    if shift == sums[i] - new_shift:
                        has_zero = True
                    sums[j] = sums[i] - new_shift
                    j += 1
            if has_zero:
                result.append(new_shift)
            else:
                result.append(-new_shift)
                shift -= new_shift
            l //= 2
        return result
import collections

class Solution2(object):

    def recoverArray(self, n, sums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type sums: List[int]\n        :rtype: List[int]\n        '
        (min_sum, max_sum) = (min(sums), max(sums))
        dp = [0] * (max_sum - min_sum + 1)
        for x in sums:
            dp[x - min_sum] += 1
        sorted_sums = [x for x in xrange(min_sum, max_sum + 1) if dp[x - min_sum]]
        shift = 0
        result = []
        for _ in xrange(n):
            new_dp = [0] * (max_sum - min_sum + 1)
            new_sorted_sums = []
            new_shift = sorted_sums[0] - sorted_sums[1] if dp[sorted_sums[0] - min_sum] == 1 else 0
            assert new_shift <= 0
            for x in sorted_sums:
                if not dp[x - min_sum]:
                    continue
                dp[x - new_shift - min_sum] -= dp[x - min_sum] if new_shift else dp[x - min_sum] // 2
                new_dp[x - new_shift - min_sum] = dp[x - min_sum]
                new_sorted_sums.append(x - new_shift)
            dp = new_dp
            sorted_sums = new_sorted_sums
            if dp[shift - min_sum]:
                result.append(new_shift)
            else:
                result.append(-new_shift)
                shift -= new_shift
        return result
import collections
import operator

class Solution3(object):

    def recoverArray(self, n, sums):
        if False:
            return 10
        '\n        :type n: int\n        :type sums: List[int]\n        :rtype: List[int]\n        '
        dp = {k: v for (k, v) in collections.Counter(sums).iteritems()}
        total = reduce(operator.ior, dp.itervalues(), 0)
        basis = total & -total
        if basis > 1:
            for k in dp.iterkeys():
                dp[k] //= basis
        sorted_sums = sorted(dp.iterkeys())
        shift = 0
        result = [0] * (basis.bit_length() - 1)
        for _ in xrange(n - len(result)):
            new_dp = {}
            new_sorted_sums = []
            new_shift = sorted_sums[0] - sorted_sums[1]
            assert new_shift < 0
            for x in sorted_sums:
                if not dp[x]:
                    continue
                dp[x - new_shift] -= dp[x]
                new_dp[x - new_shift] = dp[x]
                new_sorted_sums.append(x - new_shift)
            dp = new_dp
            sorted_sums = new_sorted_sums
            if shift in dp:
                result.append(new_shift)
            else:
                result.append(-new_shift)
                shift -= new_shift
        return result
import collections

class Solution4(object):

    def recoverArray(self, n, sums):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type sums: List[int]\n        :rtype: List[int]\n        '
        dp = {k: v for (k, v) in collections.Counter(sums).iteritems()}
        sorted_sums = sorted(dp.iterkeys())
        shift = 0
        result = []
        for _ in xrange(n):
            new_dp = {}
            new_sorted_sums = []
            new_shift = sorted_sums[0] - sorted_sums[1] if dp[sorted_sums[0]] == 1 else 0
            assert new_shift <= 0
            for x in sorted_sums:
                if not dp[x]:
                    continue
                dp[x - new_shift] -= dp[x] if new_shift else dp[x] // 2
                new_dp[x - new_shift] = dp[x]
                new_sorted_sums.append(x - new_shift)
            dp = new_dp
            sorted_sums = new_sorted_sums
            if shift in dp:
                result.append(new_shift)
            else:
                result.append(-new_shift)
                shift -= new_shift
        return result
import collections

class Solution5(object):

    def recoverArray(self, n, sums):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type sums: List[int]\n        :rtype: List[int]\n        '
        dp = OrderedDict(sorted(collections.Counter(sums).iteritems()))
        shift = 0
        result = []
        for _ in xrange(n):
            new_dp = OrderedDict()
            it = iter(dp)
            min_sum = next(it)
            new_shift = min_sum - next(it) if dp[min_sum] == 1 else 0
            assert new_shift <= 0
            for x in dp.iterkeys():
                if not dp[x]:
                    continue
                dp[x - new_shift] -= dp[x] if new_shift else dp[x] // 2
                new_dp[x - new_shift] = dp[x]
            dp = new_dp
            if shift in dp:
                result.append(new_shift)
            else:
                result.append(-new_shift)
                shift -= new_shift
        return result