class Solution(object):

    def countCombinations(self, pieces, positions):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type pieces: List[str]\n        :type positions: List[List[int]]\n        :rtype: int\n        '
        directions = {'rook': [(0, 1), (1, 0), (0, -1), (-1, 0)], 'bishop': [(1, 1), (1, -1), (-1, 1), (-1, -1)], 'queen': [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]}
        all_mask = 2 ** 7 - 1

        def backtracking(pieces, positions, i, lookup):
            if False:
                for i in range(10):
                    print('nop')
            if i == len(pieces):
                return 1
            result = 0
            (r, c) = positions[i]
            (r, c) = (r - 1, c - 1)
            mask = all_mask
            if not lookup[r][c] & mask:
                lookup[r][c] += mask
                result += backtracking(pieces, positions, i + 1, lookup)
                lookup[r][c] -= mask
            for (dr, dc) in directions[pieces[i]]:
                (bit, nr, nc) = (1, r + dr, c + dc)
                mask = all_mask
                while 0 <= nr < 8 and 0 <= nc < 8 and (not lookup[nr][nc] & bit):
                    lookup[nr][nc] += bit
                    mask -= bit
                    if not lookup[nr][nc] & mask:
                        lookup[nr][nc] += mask
                        result += backtracking(pieces, positions, i + 1, lookup)
                        lookup[nr][nc] -= mask
                    (bit, nr, nc) = (bit << 1, nr + dr, nc + dc)
                while bit >> 1:
                    (bit, nr, nc) = (bit >> 1, nr - dr, nc - dc)
                    lookup[nr][nc] -= bit
            return result
        return backtracking(pieces, positions, 0, [[0] * 8 for _ in range(8)])