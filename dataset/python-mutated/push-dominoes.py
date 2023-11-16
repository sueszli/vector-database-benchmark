class Solution(object):

    def pushDominoes(self, dominoes):
        if False:
            while True:
                i = 10
        '\n        :type dominoes: str\n        :rtype: str\n        '
        force = [0] * len(dominoes)
        f = 0
        for i in xrange(len(dominoes)):
            if dominoes[i] == 'R':
                f = len(dominoes)
            elif dominoes[i] == 'L':
                f = 0
            else:
                f = max(f - 1, 0)
            force[i] += f
        f = 0
        for i in reversed(xrange(len(dominoes))):
            if dominoes[i] == 'L':
                f = len(dominoes)
            elif dominoes[i] == 'R':
                f = 0
            else:
                f = max(f - 1, 0)
            force[i] -= f
        return ''.join(('.' if f == 0 else 'R' if f > 0 else 'L' for f in force))