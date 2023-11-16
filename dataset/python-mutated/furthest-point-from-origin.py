class Solution(object):

    def furthestDistanceFromOrigin(self, moves):
        if False:
            while True:
                i = 10
        '\n        :type moves: str\n        :rtype: int\n        '
        curr = cnt = 0
        for x in moves:
            if x == 'L':
                curr -= 1
            elif x == 'R':
                curr += 1
            else:
                cnt += 1
        return abs(curr) + cnt