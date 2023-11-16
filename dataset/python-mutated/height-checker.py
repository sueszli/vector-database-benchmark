import itertools

class Solution(object):

    def heightChecker(self, heights):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type heights: List[int]\n        :rtype: int\n        '
        return sum((i != j for (i, j) in itertools.izip(heights, sorted(heights))))