import collections
import itertools

class Solution(object):

    def timeTaken(self, arrival, state):
        if False:
            i = 10
            return i + 15
        '\n        :type arrival: List[int]\n        :type state: List[int]\n        :rtype: List[int]\n        '

        def go_until(t):
            if False:
                while True:
                    i = 10
            while curr[0] <= t and any(q):
                if not q[direction[0]]:
                    direction[0] ^= 1
                result[q[direction[0]].popleft()] = curr[0]
                curr[0] += 1
        (UNKNOWN, ENTERING, EXITING) = range(-1, 1 + 1)
        result = [0] * len(arrival)
        (curr, direction) = ([float('-inf')], [UNKNOWN])
        q = [collections.deque(), collections.deque()]
        for (i, (a, s)) in enumerate(itertools.izip(arrival, state)):
            go_until(a - 1)
            q[s].append(i)
            if not a <= curr[0]:
                (curr, direction) = ([a], [EXITING])
        go_until(float('inf'))
        return result