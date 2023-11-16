class Solution(object):

    def canChange(self, start, target):
        if False:
            i = 10
            return i + 15
        '\n        :type start: str\n        :type target: str\n        :rtype: bool\n        '
        i = j = 0
        while True:
            while i < len(start) and start[i] == '_':
                i += 1
            while j < len(target) and target[j] == '_':
                j += 1
            if i == len(start) and j == len(target):
                break
            if i == len(start) or j == len(target) or start[i] != target[j] or (start[i] == 'L' and i < j) or (start[i] == 'R' and i > j):
                return False
            i += 1
            j += 1
        return True