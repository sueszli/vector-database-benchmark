class Solution(object):

    def numberOfPatterns(self, m, n):
        if False:
            i = 10
            return i + 15
        '\n        :type m: int\n        :type n: int\n        :rtype: int\n        '

        def merge(used, i):
            if False:
                print('Hello World!')
            return used | 1 << i

        def number_of_keys(i):
            if False:
                return 10
            number = 0
            while i > 0:
                i &= i - 1
                number += 1
            return number

        def contain(used, i):
            if False:
                return 10
            return bool(used & 1 << i)

        def convert(i, j):
            if False:
                for i in range(10):
                    print('nop')
            return 3 * i + j
        dp = [[0] * 9 for _ in xrange(1 << 9)]
        for i in xrange(9):
            dp[merge(0, i)][i] = 1
        res = 0
        for used in xrange(len(dp)):
            number = number_of_keys(used)
            if number > n:
                continue
            for i in xrange(9):
                if not contain(used, i):
                    continue
                if m <= number <= n:
                    res += dp[used][i]
                (x1, y1) = divmod(i, 3)
                for j in xrange(9):
                    if contain(used, j):
                        continue
                    (x2, y2) = divmod(j, 3)
                    if (x1 == x2 and abs(y1 - y2) == 2 or (y1 == y2 and abs(x1 - x2) == 2) or (abs(x1 - x2) == 2 and abs(y1 - y2) == 2)) and (not contain(used, convert((x1 + x2) // 2, (y1 + y2) // 2))):
                        continue
                    dp[merge(used, j)][j] += dp[used][i]
        return res

class Solution2(object):

    def numberOfPatterns(self, m, n):
        if False:
            i = 10
            return i + 15
        '\n        :type m: int\n        :type n: int\n        :rtype: int\n        '

        def merge(used, i):
            if False:
                i = 10
                return i + 15
            return used | 1 << i

        def number_of_keys(i):
            if False:
                for i in range(10):
                    print('nop')
            number = 0
            while i > 0:
                i &= i - 1
                number += 1
            return number

        def exclude(used, i):
            if False:
                while True:
                    i = 10
            return used & ~(1 << i)

        def contain(used, i):
            if False:
                for i in range(10):
                    print('nop')
            return bool(used & 1 << i)

        def convert(i, j):
            if False:
                i = 10
                return i + 15
            return 3 * i + j
        dp = [[0] * 9 for _ in xrange(1 << 9)]
        for i in xrange(9):
            dp[merge(0, i)][i] = 1
        res = 0
        for used in xrange(len(dp)):
            number = number_of_keys(used)
            if number > n:
                continue
            for i in xrange(9):
                if not contain(used, i):
                    continue
                (x1, y1) = divmod(i, 3)
                for j in xrange(9):
                    if i == j or not contain(used, j):
                        continue
                    (x2, y2) = divmod(j, 3)
                    if (x1 == x2 and abs(y1 - y2) == 2 or (y1 == y2 and abs(x1 - x2) == 2) or (abs(x1 - x2) == 2 and abs(y1 - y2) == 2)) and (not contain(used, convert((x1 + x2) // 2, (y1 + y2) // 2))):
                        continue
                    dp[used][i] += dp[exclude(used, i)][j]
                if m <= number <= n:
                    res += dp[used][i]
        return res

class Solution_TLE(object):

    def numberOfPatterns(self, m, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type m: int\n        :type n: int\n        :rtype: int\n        '

        def merge(used, i):
            if False:
                for i in range(10):
                    print('nop')
            return used | 1 << i

        def contain(used, i):
            if False:
                print('Hello World!')
            return bool(used & 1 << i)

        def convert(i, j):
            if False:
                while True:
                    i = 10
            return 3 * i + j

        def numberOfPatternsHelper(m, n, level, used, i):
            if False:
                i = 10
                return i + 15
            number = 0
            if level > n:
                return number
            if m <= level <= n:
                number += 1
            (x1, y1) = divmod(i, 3)
            for j in xrange(9):
                if contain(used, j):
                    continue
                (x2, y2) = divmod(j, 3)
                if (x1 == x2 and abs(y1 - y2) == 2 or (y1 == y2 and abs(x1 - x2) == 2) or (abs(x1 - x2) == 2 and abs(y1 - y2) == 2)) and (not contain(used, convert((x1 + x2) // 2, (y1 + y2) // 2))):
                    continue
                number += numberOfPatternsHelper(m, n, level + 1, merge(used, j), j)
            return number
        number = 0
        number += 4 * numberOfPatternsHelper(m, n, 1, merge(0, 0), 0)
        number += 4 * numberOfPatternsHelper(m, n, 1, merge(0, 1), 1)
        number += numberOfPatternsHelper(m, n, 1, merge(0, 4), 4)
        return number