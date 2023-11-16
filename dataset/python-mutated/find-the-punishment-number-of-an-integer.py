class Solution(object):

    def punishmentNumber(self, n):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :rtype: int\n        '

        def backtracking(curr, target):
            if False:
                for i in range(10):
                    print('nop')
            if target == 0:
                return curr == 0
            base = 10
            while curr >= base // 10:
                (q, r) = divmod(curr, base)
                if target - r < 0:
                    break
                if backtracking(q, target - r):
                    return True
                base *= 10
            return False
        return sum((i ** 2 for i in xrange(1, n + 1) if backtracking(i ** 2, i)))