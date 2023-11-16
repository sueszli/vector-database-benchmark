import bisect

class Solution(object):

    def maximumBeauty(self, items, queries):
        if False:
            return 10
        '\n        :type items: List[List[int]]\n        :type queries: List[int]\n        :rtype: List[int]\n        '
        items.sort()
        for i in xrange(len(items) - 1):
            items[i + 1][1] = max(items[i + 1][1], items[i][1])
        result = []
        for q in queries:
            i = bisect.bisect_left(items, [q + 1])
            result.append(items[i - 1][1] if i else 0)
        return result