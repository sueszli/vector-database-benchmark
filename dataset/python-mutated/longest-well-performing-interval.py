class Solution(object):

    def longestWPI(self, hours):
        if False:
            return 10
        '\n        :type hours: List[int]\n        :rtype: int\n        '
        (result, accu) = (0, 0)
        lookup = {}
        for (i, h) in enumerate(hours):
            accu = accu + 1 if h > 8 else accu - 1
            if accu > 0:
                result = i + 1
            elif accu - 1 in lookup:
                result = max(result, i - lookup[accu - 1])
            lookup.setdefault(accu, i)
        return result