class Solution(object):

    def alphabetBoardPath(self, target):
        if False:
            i = 10
            return i + 15
        '\n        :type target: str\n        :rtype: str\n        '
        (x, y) = (0, 0)
        result = []
        for c in target:
            (y1, x1) = divmod(ord(c) - ord('a'), 5)
            result.append('U' * max(y - y1, 0))
            result.append('L' * max(x - x1, 0))
            result.append('R' * max(x1 - x, 0))
            result.append('D' * max(y1 - y, 0))
            result.append('!')
            (x, y) = (x1, y1)
        return ''.join(result)