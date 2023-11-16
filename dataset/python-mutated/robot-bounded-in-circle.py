class Solution(object):

    def isRobotBounded(self, instructions):
        if False:
            return 10
        '\n        :type instructions: str\n        :rtype: bool\n        '
        directions = [[1, 0], [0, -1], [-1, 0], [0, 1]]
        (x, y, i) = (0, 0, 0)
        for instruction in instructions:
            if instruction == 'R':
                i = (i + 1) % 4
            elif instruction == 'L':
                i = (i - 1) % 4
            else:
                x += directions[i][0]
                y += directions[i][1]
        return x == 0 and y == 0 or i > 0