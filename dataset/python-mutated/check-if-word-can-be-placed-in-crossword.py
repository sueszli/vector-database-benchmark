class Solution(object):

    def placeWordInCrossword(self, board, word):
        if False:
            while True:
                i = 10
        '\n        :type board: List[List[str]]\n        :type word: str\n        :rtype: bool\n        '

        def get_val(mat, i, j, transposed):
            if False:
                i = 10
                return i + 15
            return mat[i][j] if not transposed else mat[j][i]

        def get_vecs(mat, transposed):
            if False:
                for i in range(10):
                    print('nop')
            for i in xrange(len(mat) if not transposed else len(mat[0])):
                yield (get_val(mat, i, j, transposed) for j in xrange(len(mat[0]) if not transposed else len(mat)))
        for direction in (lambda x: iter(x), reversed):
            for transposed in xrange(2):
                for row in get_vecs(board, transposed):
                    (it, matched) = (direction(word), True)
                    for c in row:
                        if c == '#':
                            if next(it, None) is None and matched:
                                return True
                            (it, matched) = (direction(word), True)
                            continue
                        if not matched:
                            continue
                        nc = next(it, None)
                        matched = nc is not None and c in (nc, ' ')
                    if next(it, None) is None and matched:
                        return True
        return False

class Solution2(object):

    def placeWordInCrossword(self, board, word):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type board: List[List[str]]\n        :type word: str\n        :rtype: bool\n        '
        words = [word, word[::-1]]
        for mat in (board, zip(*board)):
            for row in mat:
                blocks = ''.join(row).split('#')
                for s in blocks:
                    if len(s) != len(word):
                        continue
                    for w in words:
                        if all((s[i] in (w[i], ' ') for i in xrange(len(s)))):
                            return True
        return False