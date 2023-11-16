class Solution(object):

    def calculateTax(self, brackets, income):
        if False:
            print('Hello World!')
        '\n        :type brackets: List[List[int]]\n        :type income: int\n        :rtype: float\n        '
        result = prev = 0
        for (u, p) in brackets:
            result += max((min(u, income) - prev) * p / 100.0, 0.0)
            prev = u
        return result