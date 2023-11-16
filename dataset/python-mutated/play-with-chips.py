class Solution(object):

    def minCostToMoveChips(self, chips):
        if False:
            i = 10
            return i + 15
        '\n        :type chips: List[int]\n        :rtype: int\n        '
        count = [0] * 2
        for p in chips:
            count[p % 2] += 1
        return min(count)