class Solution(object):

    def spiralMatrixIII(self, R, C, r0, c0):
        if False:
            while True:
                i = 10
        '\n        :type R: int\n        :type C: int\n        :type r0: int\n        :type c0: int\n        :rtype: List[List[int]]\n        '
        (r, c) = (r0, c0)
        result = [[r, c]]
        (x, y, n, i) = (0, 1, 0, 0)
        while len(result) < R * C:
            (r, c, i) = (r + x, c + y, i + 1)
            if 0 <= r < R and 0 <= c < C:
                result.append([r, c])
            if i == n // 2 + 1:
                (x, y, n, i) = (y, -x, n + 1, 0)
        return result