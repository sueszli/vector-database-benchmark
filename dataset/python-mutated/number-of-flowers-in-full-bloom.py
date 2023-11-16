import bisect

class Solution(object):

    def fullBloomFlowers(self, flowers, persons):
        if False:
            while True:
                i = 10
        '\n        :type flowers: List[List[int]]\n        :type persons: List[int]\n        :rtype: List[int]\n        '
        cnt = collections.Counter()
        for (s, e) in flowers:
            cnt[s] += 1
            cnt[e + 1] -= 1
        events = sorted(cnt.iterkeys())
        prefix = [0]
        for x in events:
            prefix.append(prefix[-1] + cnt[x])
        return [prefix[bisect.bisect_right(events, t)] for t in persons]
import bisect

class Solution(object):

    def fullBloomFlowers(self, flowers, persons):
        if False:
            while True:
                i = 10
        '\n        :type flowers: List[List[int]]\n        :type persons: List[int]\n        :rtype: List[int]\n        '
        (starts, ends) = ([], [])
        for (s, e) in flowers:
            starts.append(s)
            ends.append(e + 1)
        starts.sort()
        ends.sort()
        return [bisect.bisect_right(starts, t) - bisect.bisect_right(ends, t) for t in persons]