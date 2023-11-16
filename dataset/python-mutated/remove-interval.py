class Solution(object):

    def removeInterval(self, intervals, toBeRemoved):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type intervals: List[List[int]]\n        :type toBeRemoved: List[int]\n        :rtype: List[List[int]]\n        '
        (A, B) = toBeRemoved
        return [[x, y] for (a, b) in intervals for (x, y) in ((a, min(A, b)), (max(a, B), b)) if x < y]