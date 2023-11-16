class Solution(object):

    def tourOfKnight(self, m, n, r, c):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type m: int\n        :type n: int\n        :type r: int\n        :type c: int\n        :rtype: List[List[int]]\n        '
        DIRECTIONS = ((1, 2), (-1, 2), (1, -2), (-1, -2), (2, 1), (-2, 1), (2, -1), (-2, -1))

        def backtracking(r, c, i):
            if False:
                i = 10
                return i + 15

            def degree(x):
                if False:
                    print('Hello World!')
                cnt = 0
                (r, c) = x
                for (dr, dc) in DIRECTIONS:
                    (nr, nc) = (r + dr, c + dc)
                    if 0 <= nr < m and 0 <= nc < n and (result[nr][nc] == -1):
                        cnt += 1
                return cnt
            if i == m * n:
                return True
            candidates = []
            for (dr, dc) in DIRECTIONS:
                (nr, nc) = (r + dr, c + dc)
                if 0 <= nr < m and 0 <= nc < n and (result[nr][nc] == -1):
                    candidates.append((nr, nc))
            for (nr, nc) in sorted(candidates, key=degree):
                result[nr][nc] = i
                if backtracking(nr, nc, i + 1):
                    return True
                result[nr][nc] = -1
            return False
        result = [[-1] * n for _ in xrange(m)]
        result[r][c] = 0
        backtracking(r, c, 1)
        return result

class Solution2(object):

    def tourOfKnight(self, m, n, r, c):
        if False:
            print('Hello World!')
        '\n        :type m: int\n        :type n: int\n        :type r: int\n        :type c: int\n        :rtype: List[List[int]]\n        '
        DIRECTIONS = ((1, 2), (-1, 2), (1, -2), (-1, -2), (2, 1), (-2, 1), (2, -1), (-2, -1))

        def backtracking(r, c, i):
            if False:
                for i in range(10):
                    print('nop')
            if i == m * n:
                return True
            for (dr, dc) in DIRECTIONS:
                (nr, nc) = (r + dr, c + dc)
                if not (0 <= nr < m and 0 <= nc < n and (result[nr][nc] == -1)):
                    continue
                result[nr][nc] = i
                if backtracking(nr, nc, i + 1):
                    return True
                result[nr][nc] = -1
            return False
        result = [[-1] * n for _ in xrange(m)]
        result[r][c] = 0
        backtracking(r, c, 1)
        return result