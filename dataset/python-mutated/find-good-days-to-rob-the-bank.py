class Solution(object):

    def goodDaysToRobBank(self, security, time):
        if False:
            return 10
        '\n        :type security: List[int]\n        :type time: int\n        :rtype: List[int]\n        '
        right = [0]
        for i in reversed(xrange(1, len(security))):
            right.append(right[-1] + 1 if security[i] >= security[i - 1] else 0)
        right.reverse()
        result = []
        left = 0
        for i in xrange(len(security)):
            if left >= time and right[i] >= time:
                result.append(i)
            if i + 1 < len(security):
                left = left + 1 if security[i] >= security[i + 1] else 0
        return result