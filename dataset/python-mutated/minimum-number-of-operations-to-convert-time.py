class Solution(object):

    def convertTime(self, current, correct):
        if False:
            return 10
        '\n        :type current: str\n        :type correct: str\n        :rtype: int\n        '
        OPS = (60, 15, 5, 1)
        diff = int(correct[:2]) * 60 + int(correct[3:]) - (int(current[:2]) * 60 + int(current[3:]))
        result = 0
        for x in OPS:
            (q, diff) = divmod(diff, x)
            result += q
        return result