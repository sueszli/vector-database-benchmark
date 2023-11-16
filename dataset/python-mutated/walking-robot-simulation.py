class Solution(object):

    def robotSim(self, commands, obstacles):
        if False:
            i = 10
            return i + 15
        '\n        :type commands: List[int]\n        :type obstacles: List[List[int]]\n        :rtype: int\n        '
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        (x, y, i) = (0, 0, 0)
        lookup = set(map(tuple, obstacles))
        result = 0
        for cmd in commands:
            if cmd == -2:
                i = (i - 1) % 4
            elif cmd == -1:
                i = (i + 1) % 4
            else:
                for k in xrange(cmd):
                    if (x + directions[i][0], y + directions[i][1]) not in lookup:
                        x += directions[i][0]
                        y += directions[i][1]
                        result = max(result, x * x + y * y)
        return result