class Solution(object):

    def colorRed(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: List[List[int]]\n        '
        result = [[1, 1]]
        for i in xrange(2, n + 1):
            if i % 2 == n % 2:
                result.extend(([i, j] for j in xrange(1 if i % 4 == n % 4 else 3, 2 * i, 2)))
            else:
                result.append([i, 2 if i % 4 == (n - 1) % 4 else 1])
        return result