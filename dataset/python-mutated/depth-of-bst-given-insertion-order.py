import sortedcontainers

class Solution(object):

    def maxDepthBST(self, order):
        if False:
            print('Hello World!')
        '\n        :type order: List[int]\n        :rtype: int\n        '
        depths = sortedcontainers.SortedDict({float('-inf'): 0, float('inf'): 0})
        values_view = depths.values()
        result = 0
        for x in order:
            i = depths.bisect_right(x)
            depths[x] = max(values_view[i - 1:i + 1]) + 1
            result = max(result, depths[x])
        return result