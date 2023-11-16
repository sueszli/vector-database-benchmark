class Solution(object):

    def maximumEvenSplit(self, finalSum):
        if False:
            return 10
        '\n        :type finalSum: int\n        :rtype: List[int]\n        '
        if finalSum % 2:
            return []
        result = []
        i = 2
        while i <= finalSum:
            result.append(i)
            finalSum -= i
            i += 2
        result[-1] += finalSum
        return result