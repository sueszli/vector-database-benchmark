class Solution(object):

    def maxSatisfaction(self, satisfaction):
        if False:
            i = 10
            return i + 15
        '\n        :type satisfaction: List[int]\n        :rtype: int\n        '
        satisfaction.sort(reverse=True)
        (result, curr) = (0, 0)
        for x in satisfaction:
            curr += x
            if curr <= 0:
                break
            result += curr
        return result