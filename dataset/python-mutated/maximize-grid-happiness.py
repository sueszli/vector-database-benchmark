class Solution(object):

    def getMaxGridHappiness(self, m, n, introvertsCount, extrovertsCount):
        if False:
            i = 10
            return i + 15
        '\n        :type m: int\n        :type n: int\n        :type introvertsCount: int\n        :type extrovertsCount: int\n        :rtype: int\n        '

        def left(curr):
            if False:
                for i in range(10):
                    print('nop')
            return curr[-1] if len(curr) % n else 0

        def up(curr):
            if False:
                return 10
            return curr[-n] if len(curr) >= n else 0

        def count_total(curr, t, total):
            if False:
                for i in range(10):
                    print('nop')
            return total - 30 * ((left(curr) == 1) + (up(curr) == 1)) + 20 * ((left(curr) == 2) + (up(curr) == 2)) + (120 - 30 * ((left(curr) != 0) + (up(curr) != 0))) * (t == 1) + (40 + 20 * ((left(curr) != 0) + (up(curr) != 0))) * (t == 2)

        def iter_backtracking(i, e):
            if False:
                i = 10
                return i + 15
            result = 0
            curr = []
            stk = [(2, (i, e, 0))]
            while stk:
                (step, params) = stk.pop()
                if step == 2:
                    (i, e, total) = params
                    if len(curr) == m * n or (i == 0 and e == 0):
                        result = max(result, total)
                        continue
                    if total + (i + e) * 120 < result:
                        continue
                    if e > 0:
                        stk.append((3, tuple()))
                        stk.append((2, (i, e - 1, count_total(curr, 2, total))))
                        stk.append((1, (2,)))
                    if i > 0:
                        stk.append((3, tuple()))
                        stk.append((2, (i - 1, e, count_total(curr, 1, total))))
                        stk.append((1, (1,)))
                    if left(curr) or up(curr):
                        stk.append((3, tuple()))
                        stk.append((2, (i, e, total)))
                        stk.append((1, (0,)))
                elif step == 1:
                    x = params[0]
                    curr.append(x)
                elif step == 3:
                    curr.pop()
            return result
        return iter_backtracking(introvertsCount, extrovertsCount)

class Solution2(object):

    def getMaxGridHappiness(self, m, n, introvertsCount, extrovertsCount):
        if False:
            while True:
                i = 10
        '\n        :type m: int\n        :type n: int\n        :type introvertsCount: int\n        :type extrovertsCount: int\n        :rtype: int\n        '

        def left(curr):
            if False:
                return 10
            return curr[-1] if len(curr) % n else 0

        def up(curr):
            if False:
                for i in range(10):
                    print('nop')
            return curr[-n] if len(curr) >= n else 0

        def count_total(curr, t, total):
            if False:
                while True:
                    i = 10
            return total - 30 * ((left(curr) == 1) + (up(curr) == 1)) + 20 * ((left(curr) == 2) + (up(curr) == 2)) + (120 - 30 * ((left(curr) != 0) + (up(curr) != 0))) * (t == 1) + (40 + 20 * ((left(curr) != 0) + (up(curr) != 0))) * (t == 2)

        def backtracking(i, e, total, curr, result):
            if False:
                while True:
                    i = 10
            if len(curr) == m * n or (i == 0 and e == 0):
                result[0] = max(result[0], total)
                return
            if total + (i + e) * 120 < result[0]:
                return
            if left(curr) or up(curr):
                curr.append(0)
                backtracking(i, e, total, curr, result)
                curr.pop()
            if i > 0:
                new_total = count_total(curr, 1, total)
                curr.append(1)
                backtracking(i - 1, e, new_total, curr, result)
                curr.pop()
            if e > 0:
                new_total = count_total(curr, 2, total)
                curr.append(2)
                backtracking(i, e - 1, new_total, curr, result)
                curr.pop()
        result = [0]
        backtracking(introvertsCount, extrovertsCount, 0, [], result)
        return result[0]