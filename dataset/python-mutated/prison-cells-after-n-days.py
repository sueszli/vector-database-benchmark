class Solution(object):

    def prisonAfterNDays(self, cells, N):
        if False:
            while True:
                i = 10
        '\n        :type cells: List[int]\n        :type N: int\n        :rtype: List[int]\n        '
        N -= max(N - 1, 0) // 14 * 14
        for i in xrange(N):
            cells = [0] + [cells[i - 1] ^ cells[i + 1] ^ 1 for i in xrange(1, 7)] + [0]
        return cells

class Solution2(object):

    def prisonAfterNDays(self, cells, N):
        if False:
            print('Hello World!')
        '\n        :type cells: List[int]\n        :type N: int\n        :rtype: List[int]\n        '
        cells = tuple(cells)
        lookup = {}
        while N:
            lookup[cells] = N
            N -= 1
            cells = tuple([0] + [cells[i - 1] ^ cells[i + 1] ^ 1 for i in xrange(1, 7)] + [0])
            if cells in lookup:
                assert lookup[cells] - N in (1, 7, 14)
                N %= lookup[cells] - N
                break
        while N:
            N -= 1
            cells = tuple([0] + [cells[i - 1] ^ cells[i + 1] ^ 1 for i in xrange(1, 7)] + [0])
        return list(cells)