class Solution(object):

    def stoneGameVI(self, aliceValues, bobValues):
        if False:
            print('Hello World!')
        '\n        :type aliceValues: List[int]\n        :type bobValues: List[int]\n        :rtype: int\n        '
        sorted_vals = sorted(((a, b) for (a, b) in zip(aliceValues, bobValues)), key=sum, reverse=True)
        return cmp(sum((a for (a, _) in sorted_vals[::2])), sum((b for (_, b) in sorted_vals[1::2])))