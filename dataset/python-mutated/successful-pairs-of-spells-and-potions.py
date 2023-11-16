class Solution(object):

    def successfulPairs(self, spells, potions, success):
        if False:
            i = 10
            return i + 15
        '\n        :type spells: List[int]\n        :type potions: List[int]\n        :type success: int\n        :rtype: List[int]\n        '

        def ceil_divide(a, b):
            if False:
                i = 10
                return i + 15
            return (a + (b - 1)) // b
        potions.sort()
        return [len(potions) - bisect.bisect_left(potions, ceil_divide(success, s)) for s in spells]