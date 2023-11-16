class Solution(object):

    def maxDepthAfterSplit(self, seq):
        if False:
            print('Hello World!')
        '\n        :type seq: str\n        :rtype: List[int]\n        '
        return [i & 1 ^ (seq[i] == '(') for (i, c) in enumerate(seq)]

class Solution2(object):

    def maxDepthAfterSplit(self, seq):
        if False:
            i = 10
            return i + 15
        '\n        :type seq: str\n        :rtype: List[int]\n        '
        (A, B) = (0, 0)
        result = [0] * len(seq)
        for (i, c) in enumerate(seq):
            point = 1 if c == '(' else -1
            if point == 1 and A <= B or (point == -1 and A >= B):
                A += point
            else:
                B += point
                result[i] = 1
        return result