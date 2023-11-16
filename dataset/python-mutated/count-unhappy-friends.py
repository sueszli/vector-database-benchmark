class Solution(object):

    def unhappyFriends(self, n, preferences, pairs):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type preferences: List[List[int]]\n        :type pairs: List[List[int]]\n        :rtype: int\n        '
        friends = [[0] * n for _ in xrange(n)]
        for i in xrange(len(preferences)):
            for j in xrange(len(preferences[i])):
                friends[i][preferences[i][j]] = j
        pairing = [0] * n
        for (i, j) in pairs:
            (pairing[i], pairing[j]) = (j, i)
        return sum((any((friends[i][j] < friends[i][pairing[i]] and friends[j][i] < friends[j][pairing[j]] for j in xrange(len(friends[i])) if j != i and j != pairing[i])) for i in xrange(len(friends))))