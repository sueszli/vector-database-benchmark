import collections

class Solution(object):

    def findWinners(self, matches):
        if False:
            print('Hello World!')
        '\n        :type matches: List[List[int]]\n        :rtype: List[List[int]]\n        '
        lose = collections.defaultdict(int)
        players_set = set()
        for (x, y) in matches:
            lose[y] += 1
            players_set.add(x)
            players_set.add(y)
        return [[x for x in sorted(players_set) if lose[x] == i] for i in xrange(2)]