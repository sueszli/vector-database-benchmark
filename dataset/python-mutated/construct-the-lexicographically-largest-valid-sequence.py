class Solution(object):

    def constructDistancedSequence(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: List[int]\n        '

        def backtracking(n, i, result, lookup):
            if False:
                return 10
            if i == len(result):
                return True
            if result[i]:
                return backtracking(n, i + 1, result, lookup)
            for x in reversed(xrange(1, n + 1)):
                j = i if x == 1 else i + x
                if lookup[x] or j >= len(result) or result[j]:
                    continue
                (result[i], result[j], lookup[x]) = (x, x, True)
                if backtracking(n, i + 1, result, lookup):
                    return True
                (result[i], result[j], lookup[x]) = (0, 0, False)
            return False
        (result, lookup) = ([0] * (2 * n - 1), [False] * (n + 1))
        backtracking(n, 0, result, lookup)
        return result