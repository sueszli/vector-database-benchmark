class Solution(object):

    def minOperations(self, boxes):
        if False:
            i = 10
            return i + 15
        '\n        :type boxes: str\n        :rtype: List[int]\n        '
        result = [0] * len(boxes)
        for direction in (lambda x: x, reversed):
            cnt = accu = 0
            for i in direction(xrange(len(boxes))):
                result[i] += accu
                if boxes[i] == '1':
                    cnt += 1
                accu += cnt
        return result