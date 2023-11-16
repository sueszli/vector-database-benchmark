class Solution(object):

    def restoreString(self, s, indices):
        if False:
            return 10
        '\n        :type s: str\n        :type indices: List[int]\n        :rtype: str\n        '
        result = list(s)
        for (i, c) in enumerate(result):
            if indices[i] == i:
                continue
            (move, j) = (c, indices[i])
            while j != i:
                (result[j], move) = (move, result[j])
                (indices[j], j) = (j, indices[j])
            result[i] = move
        return ''.join(result)
import itertools

class Solution2(object):

    def restoreString(self, s, indices):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :type indices: List[int]\n        :rtype: str\n        '
        result = [''] * len(s)
        for (i, c) in itertools.izip(indices, s):
            result[i] = c
        return ''.join(result)