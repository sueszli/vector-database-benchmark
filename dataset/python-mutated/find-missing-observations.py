class Solution(object):

    def missingRolls(self, rolls, mean, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type rolls: List[int]\n        :type mean: int\n        :type n: int\n        :rtype: List[int]\n        '
        MAX_V = 6
        MIN_V = 1
        total = sum(rolls)
        missing = mean * (n + len(rolls)) - total
        if missing < MIN_V * n or missing > MAX_V * n:
            return []
        (q, r) = divmod(missing, n)
        return [q + int(i < r) for i in xrange(n)]

class Solution2(object):

    def missingRolls(self, rolls, mean, n):
        if False:
            while True:
                i = 10
        '\n        :type rolls: List[int]\n        :type mean: int\n        :type n: int\n        :rtype: List[int]\n        '
        MAX_V = 6
        MIN_V = 1
        total = sum(rolls)
        missing = mean * (n + len(rolls)) - total
        if missing < MIN_V * n or missing > MAX_V * n:
            return []
        (q, r) = divmod(missing - MIN_V * n, MAX_V - MIN_V)
        return [MAX_V if i < q else MIN_V + r if i == q else MIN_V for i in xrange(n)]