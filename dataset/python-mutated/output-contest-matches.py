class Solution(object):

    def findContestMatch(self, n):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :rtype: str\n        '
        matches = map(str, range(1, n + 1))
        while len(matches) / 2:
            matches = ['({},{})'.format(matches[i], matches[-i - 1]) for i in xrange(len(matches) / 2)]
        return matches[0]