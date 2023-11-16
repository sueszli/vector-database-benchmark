class Solution(object):

    def minimumCardPickup(self, cards):
        if False:
            return 10
        '\n        :type cards: List[int]\n        :rtype: int\n        '
        lookup = {}
        result = float('inf')
        for (i, x) in enumerate(cards):
            if x in lookup:
                result = min(result, i - lookup[x] + 1)
            lookup[x] = i
        return result if result != float('inf') else -1