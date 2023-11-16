import collections
import itertools

class Solution(object):

    def equalPairs(self, grid):
        if False:
            print('Hello World!')
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '
        cnt1 = collections.Counter((tuple(row) for row in grid))
        cnt2 = collections.Counter((tuple(col) for col in itertools.izip(*grid)))
        return sum((cnt1[k] * cnt2[k] for k in cnt1.iterkeys() if k in cnt2))