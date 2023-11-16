class Solution(object):

    def canTransform(self, start, end):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type start: str\n        :type end: str\n        :rtype: bool\n        '
        if start.count('X') != end.count('X'):
            return False
        (i, j) = (0, 0)
        while i < len(start) and j < len(end):
            while i < len(start) and start[i] == 'X':
                i += 1
            while j < len(end) and end[j] == 'X':
                j += 1
            if (i < len(start)) != (j < len(end)):
                return False
            elif i < len(start) and j < len(end):
                if start[i] != end[j] or (start[i] == 'L' and i < j) or (start[i] == 'R' and i > j):
                    return False
            i += 1
            j += 1
        return True