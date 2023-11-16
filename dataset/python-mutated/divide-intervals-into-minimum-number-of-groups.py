import collections

class Solution(object):

    def minGroups(self, intervals):
        if False:
            i = 10
            return i + 15
        '\n        :type intervals: List[List[int]]\n        :rtype: int\n        '
        events = collections.Counter()
        for (l, r) in intervals:
            events[l] += 1
            events[r + 1] -= 1
        result = curr = 0
        for t in sorted(events.iterkeys()):
            curr += events[t]
            result = max(result, curr)
        return result