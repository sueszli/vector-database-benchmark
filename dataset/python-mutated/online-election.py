import collections
import itertools
import bisect

class TopVotedCandidate(object):

    def __init__(self, persons, times):
        if False:
            print('Hello World!')
        '\n        :type persons: List[int]\n        :type times: List[int]\n        '
        lead = -1
        (self.__lookup, count) = ([], collections.defaultdict(int))
        for (t, p) in itertools.izip(times, persons):
            count[p] += 1
            if count[p] >= count[lead]:
                lead = p
                self.__lookup.append((t, lead))

    def q(self, t):
        if False:
            print('Hello World!')
        '\n        :type t: int\n        :rtype: int\n        '
        return self.__lookup[bisect.bisect(self.__lookup, (t, float('inf'))) - 1][1]