class Solution(object):

    def minCost(self, grid):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        def a_star(grid, b, t):
            if False:
                i = 10
                return i + 15
            (f, dh) = (0, 1)
            (closer, detour) = ([b], [])
            lookup = set()
            while closer or detour:
                if not closer:
                    f += dh
                    (closer, detour) = (detour, closer)
                b = closer.pop()
                if b in lookup:
                    continue
                lookup.add(b)
                if b == t:
                    return f
                for (nd, (dr, dc)) in enumerate(directions, 1):
                    nb = (b[0] + dr, b[1] + dc)
                    if not (0 <= nb[0] < len(grid) and 0 <= nb[1] < len(grid[0]) and (nb not in lookup)):
                        continue
                    (closer if nd == grid[b[0]][b[1]] else detour).append(nb)
            return -1
        return a_star(grid, (0, 0), (len(grid) - 1, len(grid[0]) - 1))
import collections

class Solution2(object):

    def minCost(self, grid):
        if False:
            while True:
                i = 10
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        (b, t) = ((0, 0), (len(grid) - 1, len(grid[0]) - 1))
        dq = collections.deque([(b, 0)])
        lookup = set()
        while dq:
            (b, d) = dq.popleft()
            if b in lookup:
                continue
            lookup.add(b)
            if b == t:
                return d
            for (nd, (dr, dc)) in enumerate(directions, 1):
                nb = (b[0] + dr, b[1] + dc)
                if not (0 <= nb[0] < len(grid) and 0 <= nb[1] < len(grid[0]) and (nb not in lookup)):
                    continue
                if nd == grid[b[0]][b[1]]:
                    dq.appendleft((nb, d))
                else:
                    dq.append((nb, d + 1))
        return -1