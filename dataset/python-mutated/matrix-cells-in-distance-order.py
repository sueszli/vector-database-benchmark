class Solution(object):

    def allCellsDistOrder(self, R, C, r0, c0):
        if False:
            i = 10
            return i + 15
        '\n        :type R: int\n        :type C: int\n        :type r0: int\n        :type c0: int\n        :rtype: List[List[int]]\n        '

        def append(R, C, r, c, result):
            if False:
                print('Hello World!')
            if 0 <= r < R and 0 <= c < C:
                result.append([r, c])
        result = [[r0, c0]]
        for d in xrange(1, R + C):
            append(R, C, r0 - d, c0, result)
            for x in xrange(-d + 1, d):
                append(R, C, r0 + x, c0 + abs(x) - d, result)
                append(R, C, r0 + x, c0 + d - abs(x), result)
            append(R, C, r0 + d, c0, result)
        return result