class Solution(object):

    def maxPrice(self, items, capacity):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type items: List[List[int]]\n        :type capacity: int\n        :rtype: float\n        '
        result = 0
        items.sort(key=lambda x: float(x[0]) / x[1], reverse=True)
        for (p, c) in items:
            cnt = min(c, capacity)
            capacity -= cnt
            result += float(p) / c * cnt
        return result if capacity == 0 else -1