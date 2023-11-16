import collections

class Solution(object):

    def numBusesToDestination(self, routes, S, T):
        if False:
            i = 10
            return i + 15
        '\n        :type routes: List[List[int]]\n        :type S: int\n        :type T: int\n        :rtype: int\n        '
        if S == T:
            return 0
        to_route = collections.defaultdict(set)
        for (i, route) in enumerate(routes):
            for stop in route:
                to_route[stop].add(i)
        result = 1
        q = [S]
        lookup = set([S])
        while q:
            next_q = []
            for stop in q:
                for i in to_route[stop]:
                    for next_stop in routes[i]:
                        if next_stop in lookup:
                            continue
                        if next_stop == T:
                            return result
                        next_q.append(next_stop)
                        to_route[next_stop].remove(i)
                        lookup.add(next_stop)
            q = next_q
            result += 1
        return -1