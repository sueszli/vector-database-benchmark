class Solution(object):

    def maxCount(self, banned, n, maxSum):
        if False:
            print('Hello World!')
        '\n        :type banned: List[int]\n        :type n: int\n        :type maxSum: int\n        :rtype: int\n        '
        k = min(int((-1 + (1 + 8 * maxSum)) ** 0.5 / 2), n)
        total = (k + 1) * k // 2
        result = k
        lookup = set(banned)
        for x in lookup:
            if x <= k:
                total -= x
                result -= 1
        for i in xrange(k + 1, n + 1):
            if i in lookup:
                continue
            if total + i > maxSum:
                break
            total += i
            result += 1
        return result
import bisect

class Solution2(object):

    def maxCount(self, banned, n, maxSum):
        if False:
            i = 10
            return i + 15
        '\n        :type banned: List[int]\n        :type n: int\n        :type maxSum: int\n        :rtype: int\n        '

        def check(x):
            if False:
                for i in range(10):
                    print('nop')
            return (x + 1) * x // 2 - prefix[bisect.bisect_right(sorted_banned, x)] <= maxSum
        sorted_banned = sorted(set(banned))
        prefix = [0] * (len(sorted_banned) + 1)
        for i in xrange(len(sorted_banned)):
            prefix[i + 1] = prefix[i] + sorted_banned[i]
        (left, right) = (1, n)
        while left <= right:
            mid = left + (right - left) // 2
            if not check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return right - bisect.bisect_right(sorted_banned, right)

class Solution3(object):

    def maxCount(self, banned, n, maxSum):
        if False:
            return 10
        '\n        :type banned: List[int]\n        :type n: int\n        :type maxSum: int\n        :rtype: int\n        '
        lookup = set(banned)
        result = total = 0
        for i in xrange(1, n + 1):
            if i in lookup:
                continue
            if total + i > maxSum:
                break
            total += i
            result += 1
        return result