class Solution(object):

    def rotateTheBox(self, box):
        if False:
            return 10
        '\n        :type box: List[List[str]]\n        :rtype: List[List[str]]\n        '
        result = [['.'] * len(box) for _ in xrange(len(box[0]))]
        for i in xrange(len(box)):
            k = len(box[0]) - 1
            for j in reversed(xrange(len(box[0]))):
                if box[i][j] == '.':
                    continue
                if box[i][j] == '*':
                    k = j
                result[k][-1 - i] = box[i][j]
                k -= 1
        return result