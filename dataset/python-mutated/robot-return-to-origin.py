import collections

class Solution(object):

    def judgeCircle(self, moves):
        if False:
            while True:
                i = 10
        '\n        :type moves: str\n        :rtype: bool\n        '
        count = collections.Counter(moves)
        return count['L'] == count['R'] and count['U'] == count['D']

class Solution(object):

    def judgeCircle(self, moves):
        if False:
            i = 10
            return i + 15
        '\n        :type moves: str\n        :rtype: bool\n        '
        (v, h) = (0, 0)
        for move in moves:
            if move == 'U':
                v += 1
            elif move == 'D':
                v -= 1
            elif move == 'R':
                h += 1
            elif move == 'L':
                h -= 1
        return v == 0 and h == 0