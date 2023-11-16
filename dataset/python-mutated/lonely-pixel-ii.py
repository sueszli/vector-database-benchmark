import collections

class Solution(object):

    def findBlackPixel(self, picture, N):
        if False:
            print('Hello World!')
        '\n        :type picture: List[List[str]]\n        :type N: int\n        :rtype: int\n        '
        (rows, cols) = ([0] * len(picture), [0] * len(picture[0]))
        lookup = collections.defaultdict(int)
        for i in xrange(len(picture)):
            for j in xrange(len(picture[0])):
                if picture[i][j] == 'B':
                    rows[i] += 1
                    cols[j] += 1
            lookup[tuple(picture[i])] += 1
        result = 0
        for i in xrange(len(picture)):
            if rows[i] == N and lookup[tuple(picture[i])] == N:
                for j in xrange(len(picture[0])):
                    result += picture[i][j] == 'B' and cols[j] == N
        return result

class Solution2(object):

    def findBlackPixel(self, picture, N):
        if False:
            while True:
                i = 10
        '\n        :type picture: List[List[str]]\n        :type N: int\n        :rtype: int\n        '
        lookup = collections.Counter(map(tuple, picture))
        cols = [col.count('B') for col in zip(*picture)]
        return sum((N * zip(row, cols).count(('B', N)) for (row, cnt) in lookup.iteritems() if cnt == N == row.count('B')))