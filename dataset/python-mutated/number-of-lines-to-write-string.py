class Solution(object):

    def numberOfLines(self, widths, S):
        if False:
            print('Hello World!')
        '\n        :type widths: List[int]\n        :type S: str\n        :rtype: List[int]\n        '
        result = [1, 0]
        for c in S:
            w = widths[ord(c) - ord('a')]
            result[1] += w
            if result[1] > 100:
                result[0] += 1
                result[1] = w
        return result